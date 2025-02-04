from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as GraphData
from torch_geometric.nn import (
    GCNConv, 
    global_mean_pool, 
    BatchNorm as GraphBatchNorm,
    Sequential as GraphSequential
    )

from .pretrainmodels.mae_autobin import MaeAutobin

class DeepCDR(nn.Module):
    def __init__(
        self,
        sc_foundation: MaeAutobin,
        mode: str = "classification",
        output_dim: int = 100,
        dropout_prob: float = 0.1,
        freeze_encoder: bool = True
        ):

        """
        Creates the DeepCDR model with the individual drug, gene, methylation,
        and mutation encoders.

        Parameters
        ----------
        mode: str
            The model task.
            One of [classification, regression].

        output_dim: int
            The output dimension of the individual encoders.

        dropout_prob: float
            The probability for the dropout layers of the network.
        """

        super().__init__()

        valid_modes = ["classification", "regression"]
        assert mode in valid_modes, f"mode must be one of {valid_modes}"

        self.drug_net = DrugGCN(
            output_dim=output_dim, 
            dropout_prob=dropout_prob
        )

        self.sc_encoder = scFoundationEncoder(sc_foundation)

        if freeze_encoder:
            for param in self.sc_encoder.parameters():
                param.requires_grad = False

            self.sc_encoder.eval()
        
        self.gene_net = MLP(
            input_dim=768, 
            output_dim=output_dim, 
            dropout_prob=dropout_prob
        )
        
        self.projection = nn.Sequential(
            nn.Linear(output_dim*2, 300),
            nn.Tanh(),
            nn.Dropout(p=dropout_prob),
        )

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=30, kernel_size=150, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=30, out_channels=10, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(in_channels=10, out_channels=5, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(p=dropout_prob),
            nn.Flatten(),
            nn.Dropout(p=0.2)
        )

        self.fc = nn.Linear(30, 2 if mode == "classification" else 1)

    def forward(
        self, 
        drug_graph: List[GraphData], 
        gene_expression_features: torch.Tensor,
        gene_padding: torch.Tensor,
        position_gene_ids: torch.Tensor
        ) -> torch.Tensor:

        """
        Parameters
        ----------
        drug_graph: List[GraphData]
            The graphs to be fed to the graph convolutional network.

        gene_expression_features: torch.Tensor
            The gene expression features.

        Returns
        -------
        out: torch.Tensor
            Logits if mode == classification.
            IC50 predictions if mode == regression.
        """

        drug_embedding = self.drug_net(
            drug_graph.x, 
            drug_graph.edge_index, 
            drug_graph.batch
        )

        gene_expression_embedding = self.sc_encoder(
            gene_expression_features,
            gene_padding,
            position_gene_ids
            )

        gene_expression_embedding = self.gene_net(
            gene_expression_embedding
        )

        combined_embedding = torch.cat([
            drug_embedding,
            gene_expression_embedding,
        ], dim=-1)

        x = self.projection(combined_embedding)
        x = x.unsqueeze(1)
        x = self.conv(x)
        
        out = self.fc(x)

        return out


class DrugGCN(nn.Module):
    def __init__(
        self,
        input_dim: int = 75,
        hidden_dim: int = 256,
        num_hidden: int = 3,
        output_dim: int = 100,
        dropout_prob: float = 0.1
        ):

        """
        The Graph Convolutional Network for drug encoding.

        Parameters
        ----------
        input_dim: int
            The dimension of the input features.

        hidden_dim: int
            The hidden dimension.

        num_hidden: int
            The number of hidden dimensions.

        output_dim: int
            The dimension of the output embedding.

        dropout_prob: float
            The probability of the dropout layers.
        """

        super().__init__()

        self.embedding_layer = GraphSequential("x, edge_index", [
            (GCNConv(input_dim, hidden_dim), "x, edge_index -> x"),
            nn.ReLU(),
            GraphBatchNorm(hidden_dim),
            nn.Dropout(p=dropout_prob)
        ])

        hidden_layers = []

        for _ in range(num_hidden-2):
            hidden_layers.append((GCNConv(hidden_dim, hidden_dim), "x, edge_index -> x"))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(GraphBatchNorm(hidden_dim))
            hidden_layers.append(nn.Dropout(p=dropout_prob))
        
        self.hidden_layer = GraphSequential("x, edge_index", hidden_layers)

        self.output_layer = GraphSequential("x, edge_index", [
            (GCNConv(hidden_dim, output_dim), "x, edge_index -> x"),
            nn.ReLU(),
            GraphBatchNorm(output_dim),
            nn.Dropout(p=dropout_prob)
        ])
        
    def forward(self, x, edge_index, batch):
        x = self.embedding_layer(x, edge_index)
        x = self.hidden_layer(x, edge_index)
        x = self.output_layer(x, edge_index)

        embedding = global_mean_pool(x, batch)

        return embedding
    

class scFoundationEncoder(nn.Module):
    def __init__(
        self,
        model: nn.Module
        ):
        super().__init__()

        self.token_embedder = model.token_emb
        self.position_embedder = model.pos_emb
        self.encoder = model.encoder

    def forward(
        self, 
        x, 
        x_padding,
        position_gene_ids
        ):

        x = self.token_embedder(torch.unsqueeze(x, 2).float(), output_weight=0)
        position_embedding = self.position_embedder(position_gene_ids)

        x = x + position_embedding
        gene_embedding = self.encoder(x, x_padding)

        gene_embedding, _ = torch.max(gene_embedding, dim=1)

        return gene_embedding
    

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 256,
        output_dim: int = 100,
        dropout_prob: float = 0.1
        ):

        """
        The MLP module for encoding gene expression and methylation features.

        input_dim: int
            The dimension of the input features.

        embedding_dim: int
            The dimension of the first layer.

        output_dim: int
            The dimension of the output embedding.

        dropout_prob: float
            The probability of the dropoout layers.
        """

        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.Tanh(),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(p=dropout_prob)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc1(x)
        embedding = self.fc2(x)

        return embedding