# federated/server.py

from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg

def server_fn(server_address="0.0.0.0:9091", num_rounds=3):
    """
    Flower SuperNode 기반 연합학습 서버를 실행합니다.
    """
    strategy = FedAvg(
<<<<<<< HEAD
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
=======
        fraction_fit=1.0,             # 전체 클라이언트 참여
        fraction_evaluate=1.0,            # 전체 클라이언트 평가
        min_fit_clients=3,            # 최소 학습 클라이언트 수
        min_evaluate_clients=3,           # 최소 평가 클라이언트 수
        min_available_clients=1       # 전체 최소 연결 클라이언트 수
>>>>>>> parent of 2d72652 (	modified:   federated/server.py)
    )

    # Construct ServerConfig
    config = ServerConfig(num_rounds=num_rounds)

    # Wrap everything into a `ServerAppComponents` object
    return ServerAppComponents(strategy=strategy, config=config)


server_app = ServerApp(server_fn=server_fn)
