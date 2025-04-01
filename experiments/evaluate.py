# experiments/evaluate.py

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.loader import DataLoader

from graph.dataset import FraudGraphDataset
from graph.gnn_model import GCN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"📊 Evaluation Result:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")

    return acc, prec, rec, f1


def run_eval(data_path, model_path=None):
    # 데이터셋 로드
    dataset = FraudGraphDataset(data_path)
    loader = DataLoader(dataset, batch_size=1)
    input_dim = dataset.graph_data.x.shape[1]

    # 모델 로드
    model = GCN(in_channels=input_dim).to(DEVICE)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✅ 모델 로드 완료: {model_path}")

    evaluate(model, loader)


if __name__ == "__main__":
    # 모델 경로 없으면 초기화된 GCN으로 평가됨
    run_eval("data/preprocessed/client_0.csv", model_path=None)
