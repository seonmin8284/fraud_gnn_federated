# scripts/split_clients.py

import pandas as pd
import os

os.makedirs("data/preprocessed", exist_ok=True)

train = pd.read_csv("data/train_transaction.csv")
identity = pd.read_csv("data/train_identity.csv")

merged = train.merge(identity, how="left", on="TransactionID")

# card1 기준으로 나누기 (간단히 앞에서 2개만 사용)
clients = merged["card1"].dropna().unique()
clients = clients[:2]

for i, cid in enumerate(clients):
    df = merged[merged["card1"] == cid]
    df.to_csv(f"data/preprocessed/client_{i}.csv", index=False)

print("✅ 클라이언트 데이터 분할 완료!")
