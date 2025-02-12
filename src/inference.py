import os
from typing import Tuple

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score

from utils import (
    get_args,
    save_results,
    extract_graph,
    load_model_frommmf,
    process_expression_dict,
    DeepCDR,
    MultiModalDataset
    )

@torch.no_grad()
def inference(
    dataloader: DataLoader,
    criterion: nn.Module,
    model: DeepCDR,
    mode: str,
    device: str,
    reference_path: str,
    save_dir: str
    ) -> Tuple[float, float]:

    """
    Runs inference and saves results as a csv file.

    Parameters
    ----------
    dataloader: DataLoader
        The data loader to iterate over.
    
    criterion: nn.Module
        The loss function.

    model: DeepCDR
        The model to be trained.

    mode: str
        The model task.
        One of [classification, regression].

    device: str
        One of [cuda, cpu].

    reference_path: str
        The path pointing to the test table.

    save_dir: str
        The directory to save reaults.

    Returns
    -------
    loss: float
        The average loss across the test set.

    performance: float
        The average Balaced Accuracy if mode == classification.
        The average Pearson Correlation Coefficient if mode == regression.
    """
    
    metrics = {
        "loss": [],
        "prediction": [],
        "target": [],
    }

    model.eval()
    for drug_dict, expression_dict, target in tqdm(dataloader, desc="Inference in progress"):
        drug_graphs = extract_graph(drug_dict).to(device)
        expression_dict = process_expression_dict(expression_dict)

        gene_expression_features = expression_dict["x"].to(device)
        gene_padding = expression_dict["x_padding"].to(device)
        position_gene_ids = expression_dict["position_gene_ids"].to(device)

        target = target.to(device)

        out = model(
            drug_graphs, 
            gene_expression_features,
            gene_padding,
            position_gene_ids
        )
        loss = criterion(out, target)

        if mode == "classification":
            confidence = F.softmax(out, dim=1)
            pred = torch.argmax(confidence, dim=1)

            metrics["prediction"].extend(pred.cpu().numpy())
            metrics["target"].extend(target.cpu().numpy())

        if mode == "regression":
            pred = out

            metrics["prediction"].extend(pred.cpu().numpy().squeeze(-1))
            metrics["target"].extend(target.cpu().numpy().squeeze(-1))

        metrics["loss"].extend(loss.squeeze().cpu().numpy())

    loss = sum(metrics["loss"]) / len(dataloader)
    
    if mode == "classification":
        performance = balanced_accuracy_score(metrics["target"], metrics["prediction"])

    if mode == "regression":
        performance, _ = pearsonr(metrics["target"], metrics["prediction"])

    save_results(metrics, reference_path, os.path.join(save_dir, "results.csv"))

    return loss, performance


def main():
    arg_path = os.path.join("configs", "inference.yaml")
    args = get_args(arg_path)
    identifier = args["identifier"]

    data_dir = os.path.join("..", "data", "cleaned")
    save_dir = os.path.join("..", "assets", "inference-tables", args["mode"], identifier)
    os.makedirs(save_dir, exist_ok=True)

    run_path = os.path.join("runs", args["mode"], identifier, "run-config.yaml")
    weight_path = os.path.join("..", "assets", "models", args["mode"], identifier, args["weights"])
    run_args = get_args(run_path)

    sc_pretrained_path = os.path.join("..", "assets", "models", "pre-train", "models.ckpt")
    sc_foundation, sc_config = load_model_frommmf(sc_pretrained_path)

    inference_dataset = MultiModalDataset(
        drug_dir=os.path.join(data_dir, "drugs"),
        table_path=os.path.join(data_dir, "test.csv"),
        gene_expression_path=os.path.join(data_dir, "gene-expression-normalized.csv"),
        pretrain_config=sc_config,
        mode=args["mode"]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("cuda available: ", torch.cuda.is_available())

    inference_loader = DataLoader(inference_dataset, batch_size=args["batch_size"], shuffle=False)

    model = DeepCDR(
        sc_foundation=sc_foundation,
        mode=run_args["mode"],
        output_dim=run_args["output_dim"],
        dropout_prob=run_args["dropout_prob"]
    ).to(device)

    weights = torch.load(weight_path, map_location=torch.device(device), weights_only=True)
    model.load_state_dict(weights)

    if args["mode"] == "classification":
        criterion = nn.CrossEntropyLoss(reduction="none")
        performance_metric = "Balanced Accuracy"

    if args["mode"] == "regression": 
        criterion = nn.MSELoss(reduction="none")
        performance_metric = "Pearson Correlation"

    loss, performance = inference(
        dataloader=inference_loader,
        criterion=criterion,
        model=model,
        mode=args["mode"],
        device=device,
        reference_path=os.path.join(data_dir, "test.csv"),
        save_dir=save_dir
    )

    print("\nInference Summary:")
    print(f"Loss: {loss:.4f} | {performance_metric}: {performance:.4f}\n")


if __name__ == "__main__":
    main()