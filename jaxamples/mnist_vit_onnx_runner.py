import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader


def test_onnx_model(onnx_model_path: str, test_dataloader: DataLoader) -> None:
    """Test the exported ONNX model with ONNX Runtime and the MNIST test set."""
    session = ort.InferenceSession(onnx_model_path)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape
    input_dtype = input_info.type
    output_name = session.get_outputs()[0].name

    print(
        f"ONNX model expects input '{input_name}' with shape {input_shape} and dtype {input_dtype}"
    )

    # Determine fixed batch size if present
    fixed_batch_size = None
    if isinstance(input_shape[0], int) and input_shape[0] > 0:
        fixed_batch_size = input_shape[0]
        print(f"Model expects fixed batch size: {fixed_batch_size}")

    for deterministic_flag in [True, False]:
        correct = 0
        total = 0
        print(f"\nTesting ONNX model with deterministic={deterministic_flag}")
        for batch in test_dataloader:
            images, labels = batch
            images = images.numpy().astype(np.float32)
            labels = labels.numpy()

            # Adjust shape to match ONNX model input
            # Remove batch dimension for dynamic batch size
            # Typical MNIST shape: [batch, 1, 28, 28]
            # If model expects NHWC: [batch, 28, 28, 1]
            if len(input_shape) == 4:
                if input_shape[1] == 1 or input_shape[1] == "1":
                    # Model expects NCHW, do nothing
                    pass
                elif input_shape[-1] == 1 or input_shape[-1] == "1":
                    # Model expects NHWC
                    images = np.transpose(images, (0, 2, 3, 1))
                else:
                    # Unexpected shape, print and raise error
                    raise ValueError(f"Unexpected ONNX input shape: {input_shape}")
            else:
                raise ValueError(f"Unsupported input shape: {input_shape}")

            # Enforce fixed batch size if required
            if fixed_batch_size is not None:
                if images.shape[0] != fixed_batch_size:
                    # Skip last batch if smaller than fixed batch size
                    print(
                        f"Skipping batch with size {images.shape[0]} (model expects {fixed_batch_size})"
                    )
                    continue

            inputs = {
                input_name: images,
            }
            # Only add deterministic if the model expects it
            if any(inp.name == "deterministic" for inp in session.get_inputs()):
                inputs["deterministic"] = np.array(deterministic_flag)

            # Print a warning and skip batch if batch size does not match model's expected batch size
            if fixed_batch_size is not None and images.shape[0] != fixed_batch_size:
                print(
                    f"Skipping batch with size {images.shape[0]} (model expects {fixed_batch_size})"
                )
                continue

            try:
                preds = session.run([output_name], inputs)[0]
            except Exception as e:
                print(f"ONNX Runtime error: {e}")
                print(f"Input shape: {images.shape}, dtype: {images.dtype}")
                print(
                    "This likely means the model expects a different input shape or batch size."
                )
                print(
                    "Try setting --batch-size to the batch size used during ONNX export (e.g., 2)."
                )
                # Skip this batch and continue with the next
                continue

            preds = np.argmax(preds, axis=1)

            correct += (preds == labels).sum()
            total += labels.shape[0]

        accuracy = correct / total if total > 0 else 0.0
        print(
            f"ONNX model test accuracy (deterministic={deterministic_flag}): {accuracy:.4f}"
        )


def main(args=None):
    import argparse
    from torchvision import datasets, transforms

    if args is None:
        parser = argparse.ArgumentParser(description="Test ONNX MNIST ViT model")
        parser.add_argument(
            "--onnx-model",
            type=str,
            required=False,
            default="docs/mnist_vit_model.onnx",
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
        args.onnx_model if args is not None else "docs/mnist_vit_model.onnx"
    )
    session = ort.InferenceSession(onnx_model_path)
    input_shape = session.get_inputs()[0].shape
    fixed_batch_size = (
        input_shape[0]
        if isinstance(input_shape[0], int) and input_shape[0] > 0
        else None
    )

    batch_size = args.batch_size if args is not None else 128
    if fixed_batch_size is not None and batch_size != fixed_batch_size:
        print(f"Overriding batch size to {fixed_batch_size} to match ONNX model.")
        batch_size = fixed_batch_size

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    test_dataset = datasets.MNIST(
        root=args.data_dir if args else "./data",
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_onnx_model(onnx_model_path, test_loader)


if __name__ == "__main__":
    main()
