import os
from typing import Dict, Any

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Data as GraphData
from torch_geometric.loader import DataLoader as GraphDataLoader

from .load import gatherData


class MultiModalDataset(Dataset):
    def __init__(
        self,
        drug_dir: str,
        table_path: str,
        gene_expression_path: str, # normalized
        pretrain_config: Dict[str, Any],
        mode: str = "classification"
        ):

        super().__init__()
        valid_modes = ["classification", "regression"]
        assert mode in valid_modes, f"mode must be one of {valid_modes}"

        self.drug_dir = drug_dir
        self.table = pd.read_csv(table_path)
        self.expression_df = pd.read_csv(gene_expression_path, index_col=0)
        self.pretrain_config = pretrain_config
        self.mode = mode

    def __len__(self):
        return self.table.shape[0]
    
    def __getitem__(self, idx):
        reference_row = self.table.iloc[idx]

        drug_id = str(reference_row["drug_id"])
        cell_line_id = reference_row["cell_line_id"]
        target = reference_row["label"] if self.mode == "classification" else reference_row["ic50"]

        if self.mode == "classification":
            target = torch.tensor(target, dtype=torch.long)

        else:
            target = torch.tensor(target, dtype=torch.float32).unsqueeze(-1)

        drug_dir = os.path.join(self.drug_dir, drug_id)
        drug_feature_path = os.path.join(drug_dir, "drug-feature.npy")
        drug_edge_list_path = os.path.join(drug_dir, "drug-edge-list.npy")

        expression_row = self.expression_df.loc[cell_line_id].values

        if self.pretrain_config["rawcount"] == False:
            gene_expression = torch.tensor(expression_row)

        else:
            total_count = expression_row.sum()
            gene_expression = torch.tensor(expression_row.tolist() + [total_count, total_count])

        data_gene_ids = torch.arange(19_266)
        value_labels = gene_expression > 0

        drug_dict = {
            "feature_path": drug_feature_path,
            "edge_list_path": drug_edge_list_path
        }

        expression_dict = {
            "gene_expression": gene_expression,
            "value_label": value_labels,
            "data_gene_id": data_gene_ids,
            "pad_token_id": self.pretrain_config["pad_token_id"]
        }

        return drug_dict, expression_dict, target
    
def process_expression_dict(
    expression_dict: Dict[str, Any]
    ):

    gene_expression = expression_dict["gene_expression"]
    value_labels = expression_dict["value_label"]
    data_gene_ids = expression_dict["data_gene_id"]
    pad_token_id = expression_dict["pad_token_id"][0]

    x, x_padding = gatherData(
        gene_expression, 
        value_labels, 
        pad_token_id
        )
    
    position_gene_ids, _ = gatherData(
        data_gene_ids,
        value_labels,
        pad_token_id
    )

    expression_dict = {
        "x": x,
        "x_padding": x_padding,
        "position_gene_ids": position_gene_ids
    }

    return expression_dict

class scFoundationDataset(Dataset):
    def __init__(
        self,
        gene_expression_path: str, # normalized
        pretrain_config: Dict[str, Any]
        ):
        super().__init__()

        self.gene_df = pd.read_csv(gene_expression_path)
        self.pretrain_config = pretrain_config

    def __len__(self):
        return len(self.gene_df)
    
    def __getitem__(self, idx):
        row = self.gene_df.iloc[idx, :].values

        if self.pretrain_config["rawcount"] == False:
            pretrain_gene_x = torch.tensor(row.unsqueeze(0))

        else:
            total_count = row.sum()
            pretrain_gene_x = torch.tensor(row.tolist() + [total_count, total_count]).unsqueeze(0)
        
        data_gene_ids = torch.arange(19_266).repeat(pretrain_gene_x.shape[0], 1)
        value_labels = pretrain_gene_x > 0

        x, x_padding = gatherData(
            pretrain_gene_x, 
            value_labels, 
            self.pretrain_config["pad_token_id"]
            )
        
        position_gene_ids, _ = gatherData(
            data_gene_ids,
            value_labels,
            self.pretrain_config["pad_token_id"]
        )
    
        return x, x_padding, position_gene_ids
    

def extract_graph(drug_dict: Dict[str, str]):

    """
    Extract graphs from a dictionary of graph paths.
    """

    features = [np.load(f) for f in drug_dict["feature_path"]]
    edge_lists = [np.load(f) for f in drug_dict["edge_list_path"]]

    features = [torch.tensor(f, dtype=torch.float32) for f in features]
    edge_lists = [torch.tensor(e, dtype=torch.long) for e in edge_lists]

    graphs = [GraphData(x=feature, edge_index=edge_list) for feature, edge_list in zip(features, edge_lists)]
    graph_loader = GraphDataLoader(graphs, batch_size=len(graphs), shuffle=False)
    graphs = next(iter(graph_loader))

    return graphs