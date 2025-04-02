# federated/server_main.py

from flwr.server import start_server
from federated.server import server_app

if __name__ == "__main__":
    start_server(server=server_app, config={"server_address": "0.0.0.0:9091"})
