# federated/client_main.py

import argparse
from flwr.client import ClientApp
from federated.client import client_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=str, required=True)
    args = parser.parse_args()

    app = ClientApp(lambda: client_fn(args.cid))
    app.run()

if __name__ == "__main__":
    main()
