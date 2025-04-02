# federated/client.py

from collections import OrderedDict
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from flwr.common import NDArrays, Scalar, Context
from flwr.client import NumPyClient, ClientApp

from graph.gnn_model import GCN
from graph.dataset import FraudGraphDataset
from graph.graph_utils import build_graph_from_df
from federated.utils import get_parameters, set_parameters

import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FraudClient(NumPyClient):
    def __init__(self, model, data_path: str) -> None:
        super().__init__()
        self.model = model.to(DEVICE)
        self.dataset = FraudGraphDataset(data_path)
        self.loader = DataLoader(self.dataset, batch_size=1)

    def get_parameters(self, config=None):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(1):
            for data in self.loader:
                data = data.to(DEVICE)
                optimizer.zero_grad()
                out = self.model(data.x, data.edge_index)
                loss = F.cross_entropy(out, data.y)
                loss.backward()
                optimizer.step()

        return self.get_parameters(), len(self.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        self.model.eval()

        correct = 0
        total = 0
        for data in self.loader:
            data = data.to(DEVICE)
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.y.size(0)

        acc = correct / total
        return float(acc), len(self.dataset), {"accuracy": acc}


# 👉 ClientApp 방식으로 변경
def client_fn(context: Context) -> NumPyClient:
    cid = context.node_config["partition_id"]
    data_path = f"data/preprocessed/client_{cid}.csv"

    df = pd.read_csv(data_path)
    graph_data = build_graph_from_df(df)
    input_dim = graph_data.x.shape[1]

    model = GCN(in_channels=input_dim)
    return FraudClient(model, data_path).to_client()


client_app = ClientApp(client_fn=client_fn)
