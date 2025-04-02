# federated/client.py

import flwr as fl
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from graph.gnn_model import GCN
from graph.dataset import FraudGraphDataset
from federated.utils import get_parameters, set_parameters

from flwr.client import ClientApp, NumPyClient

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FraudClient(NumPyClient):
    def __init__(self, model, data_path):
        self.model = model.to(DEVICE)
        self.dataset = FraudGraphDataset(data_path)
        self.loader = DataLoader(self.dataset, batch_size=1)

    def get_parameters(self, config=None):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(1):  # 간단하게 1 epoch
            print(f"🌀 [Client] epoch {epoch} 시작")
            for data in self.loader:
                data = data.to(DEVICE)
                optimizer.zero_grad()
                out = self.model(data.x, data.edge_index)
                loss = F.cross_entropy(out, data.y)
                loss.backward()
                optimizer.step()
            print(f"✅ [Client] epoch {epoch} 종료")
        return self.get_parameters(), len(self.dataset), {}

    def evaluate(self, parameters, config):
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


# Flower에서 호출할 client_fn
def client_fn(cid: str) -> NumPyClient:
    data_path = f"data/preprocessed/client_{cid}.csv"

    # 그래프의 input_dim 추정
    from graph.graph_utils import build_graph_from_df
    import pandas as pd

    df = pd.read_csv(data_path)
    graph_data = build_graph_from_df(df)
    input_dim = graph_data.x.shape[1]

    model = GCN(in_channels=input_dim)
    return FraudClient(model, data_path)


# ClientApp 등록 (flower-supernode CLI에서 참조됨)
app = ClientApp(client_fn)
