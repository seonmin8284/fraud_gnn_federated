# federated/client.py

import flwr as fl
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from graph.gnn_model import GCN
from graph.dataset import FraudGraphDataset
from federated.utils import get_parameters, set_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FraudClient(fl.client.NumPyClient):
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

        for epoch in range(1):  # 간단하게 1 epoch만 (config로 조정 가능)
            for data in self.loader:
                data = data.to(DEVICE)
                optimizer.zero_grad()
                out = self.model(data.x, data.edge_index)
                loss = F.cross_entropy(out, data.y)
                loss.backward()
                optimizer.step()

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


def run_client(cid, data_path, server_addr="127.0.0.1:8080"):
    # dummy data로 input_dim 추정
    from graph.graph_utils import build_graph_from_df
    import pandas as pd
    sample_df = pd.read_csv(data_path)
    sample_data = build_graph_from_df(sample_df)
    input_dim = sample_data.x.shape[1]

    model = GCN(in_channels=input_dim)

    client = FraudClient(model, data_path)
    fl.client.start_numpy_client(server_address=server_addr, client=client)
