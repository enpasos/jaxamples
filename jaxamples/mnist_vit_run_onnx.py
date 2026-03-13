from torch.utils.data import DataLoader

from jaxamples.mnist_data import get_mnist_transform
from jaxamples.mnist_onnx_eval import get_onnx_fixed_batch_size, test_onnx_model


def main(
    args=None,
    *,
    description: str = "Test ONNX MNIST model",
    default_onnx_model: str = "output/mnist_vit_model.onnx",
):
    import argparse
    from torchvision import datasets

    if args is None:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument(
            "--onnx-model",
            type=str,
            required=False,
            default=default_onnx_model,
            help="Path to ONNX model file",
        )
        parser.add_argument(
            "--batch-size", type=int, default=128, help="Batch size for DataLoader"
        )
        parser.add_argument(
            "--data-dir", type=str, default="./data", help="Directory for MNIST data"
        )
        args = parser.parse_args()

    # If ONNX model expects fixed batch size, set DataLoader batch_size accordingly
    # We'll check this after loading the model
    onnx_model_path = (
        args.onnx_model if args is not None else default_onnx_model
    )
    batch_size = args.batch_size if args is not None else 128
    fixed_batch_size = get_onnx_fixed_batch_size(onnx_model_path)
    if fixed_batch_size is not None and batch_size != fixed_batch_size:
        print(f"Overriding batch size to {fixed_batch_size} to match ONNX model.")
        batch_size = fixed_batch_size

    transform = get_mnist_transform()
    test_dataset = datasets.MNIST(
        root=args.data_dir if args else "./data",
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_onnx_model(onnx_model_path, test_loader, deterministic_flags=(True, False))


if __name__ == "__main__":
    main()
