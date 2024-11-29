from argparse import ArgumentParser

from . import run


if __name__ == "__main__":
    parser = ArgumentParser(prog="voskhttp")
    parser.add_argument("--hostname")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    run(args.hostname, args.port)
