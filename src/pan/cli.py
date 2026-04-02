import argparse

from pan.core import say_hello


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pan",
        description="Hello-world CLI for pan",
    )
    parser.add_argument(
        "--name",
        default="world",
        help="Name to greet",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print(say_hello(args.name))


if __name__ == "__main__":
    main()
