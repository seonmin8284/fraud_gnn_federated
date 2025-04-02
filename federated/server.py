# federated/server.py

from flwr.server.supernode.api import start_supernode_server
from flwr.server.strategy import FedAvg

def start_server(server_address="0.0.0.0:9091", num_rounds=3):
    """
    Flower SuperNode 기반 연합학습 서버를 실행합니다.
    """
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )

    start_supernode_server(
        server_address=server_address,
        config={"num_rounds": num_rounds},
        strategy=strategy,
    )

if __name__ == "__main__":
    start_server()
