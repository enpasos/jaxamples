import argparse
import os
from typing import Sequence

from jaxamples.mnist_config import MnistExampleConfig


def parse_example_args(
    args: Sequence[str] | None,
    *,
    description: str,
    default_onnx_output: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs to train in this run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the MNIST training batch size.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory used for MNIST downloads and caching.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory used for saving and resuming checkpoints.",
    )
    parser.add_argument(
        "--onnx-output",
        type=str,
        default=default_onnx_output,
        help="Where to save the exported ONNX model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for plots and metrics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the top-level RNG seed.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the training loop and only run export/evaluation.",
    )
    parsed_args = parser.parse_args(args)
    parsed_args.default_onnx_output = default_onnx_output
    return parsed_args


def apply_common_overrides(
    config: MnistExampleConfig, args: argparse.Namespace
) -> MnistExampleConfig:
    if args.seed is not None:
        config.seed = args.seed
    if args.epochs is not None:
        config.training.num_epochs_to_train_now = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.data_dir is not None:
        config.training.data_dir = args.data_dir
    if args.checkpoint_dir is not None:
        config.training.checkpoint_dir = args.checkpoint_dir
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir

    config.training.checkpoint_dir = os.path.abspath(config.training.checkpoint_dir)
    if config.training.output_dir is not None:
        config.training.output_dir = os.path.abspath(config.training.output_dir)

    config.onnx.output_path = os.path.abspath(args.onnx_output)

    if args.skip_training:
        config.training.enable_training = False
    config.validate()
    return config
