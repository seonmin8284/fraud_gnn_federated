# federated/client_main.py

import argparse
import flwr as fl
from flwr.common import Context
from federated.client import client_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=str, required=True)
    args = parser.parse_args()

    # node_config 안에 partition_id 포함시켜서 context로 전달
    fl.client.start_client(
        client=client_app,
        server_address="localhost:8080",
        config={"node_config": {"partition_id": args.cid}},
    )


if __name__ == "__main__":
    main()
