# federated/server.py

import flwr as fl
from flwr.server.strategy import FedAvg

def start_server(server_address="0.0.0.0:8080", num_rounds=3):
    """
    연합학습 서버를 실행합니다.
    """
    strategy = FedAvg(
        fraction_fit=1.0,             # 전체 클라이언트 참여
        fraction_evaluate=1.0,            # 전체 클라이언트 평가
        min_fit_clients=2,            # 최소 학습 클라이언트 수
        min_evaluate_clients=2,           # 최소 평가 클라이언트 수
        min_available_clients=2       # 전체 최소 연결 클라이언트 수
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )

if __name__ == "__main__":
    start_server()
