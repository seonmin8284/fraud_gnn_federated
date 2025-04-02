# federated/server.py

from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg

def server_fn(server_address="0.0.0.0:9091", num_rounds=3):
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

    # Construct ServerConfig
    config = ServerConfig(num_rounds=num_rounds)

    # Wrap everything into a `ServerAppComponents` object
    return ServerAppComponents(strategy=strategy, config=config)


server_app = ServerApp(server_fn=server_fn)
