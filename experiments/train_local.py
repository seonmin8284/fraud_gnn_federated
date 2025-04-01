# experiments/train_local.py

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from graph.dataset import FraudGraphDataset
from graph.gnn_model import GCN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, loader, optimizer):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss

def test(model, loader):
    model.eval()
    correct = 0
    total = 0

    for data in loader:
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        total += data.y.size(0)

    return correct / total


def run_local_train(data_path, epochs=5):
    dataset = FraudGraphDataset(data_path)
    loader = DataLoader(dataset, batch_size=1)

    input_dim = dataset.graph_data.x.shape[1]
    model = GCN(in_channels=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        loss = train(model, loader, optimizer)
        acc = test(model, loader)
        print(f"[Epoch {epoch+1}] Loss: {loss:.4f}, Accuracy: {acc:.4f}")


if __name__ == "__main__":
    # client_0과 client_1을 합친 중앙 데이터셋으로도 가능
    run_local_train("data/preprocessed/client_0.csv")
