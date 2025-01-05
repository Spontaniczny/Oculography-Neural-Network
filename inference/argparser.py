import argparse

def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--save_width",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--save_height",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--save_folder",
        type=str,
        default="1",
    )

    parser.add_argument(
        "--remove_artifacts",
        type=bool,
        default=False
    )

    parser.add_argument(
        "--save_step",
        type=int,
        default=0,
        help="One of every save_step images is saved"
    )

    return parser