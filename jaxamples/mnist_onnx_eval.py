from typing import Any, Iterable, Sequence

import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader


def get_onnx_fixed_batch_size(onnx_model_path: str) -> int | None:
    session = ort.InferenceSession(onnx_model_path)
    input_shape = session.get_inputs()[0].shape
    return input_shape[0] if isinstance(input_shape[0], int) and input_shape[0] > 0 else None


def prepare_onnx_images(
    images_nchw: np.ndarray,
    *,
    input_shape: Sequence[Any],
) -> np.ndarray:
    if len(input_shape) != 4:
        raise ValueError(f"Unsupported input shape: {input_shape}")

    nchw_tail = images_nchw.shape[1:]
    nhwc_tail = (images_nchw.shape[2], images_nchw.shape[3], images_nchw.shape[1])

    if tuple(input_shape[1:]) == nchw_tail:
        return images_nchw
    if tuple(input_shape[1:]) == nhwc_tail:
        return np.transpose(images_nchw, (0, 2, 3, 1))
    raise ValueError(f"Unexpected ONNX input shape: {input_shape}")


def test_onnx_model(
    onnx_model_path: str,
    test_dataloader: DataLoader,
    *,
    deterministic_flags: Iterable[bool] = (True,),
) -> None:
    """Test an exported ONNX model against a MNIST-style dataloader."""
    session = ort.InferenceSession(onnx_model_path)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape
    input_dtype = input_info.type
    output_name = session.get_outputs()[0].name
    deterministic_input = next(
        (inp.name for inp in session.get_inputs() if inp.name == "deterministic"),
        None,
    )
    fixed_batch_size = (
        input_shape[0] if isinstance(input_shape[0], int) and input_shape[0] > 0 else None
    )

    print(
        f"ONNX model expects input '{input_name}' with shape {input_shape} and dtype {input_dtype}"
    )
    if fixed_batch_size is not None:
        print(f"Model expects fixed batch size: {fixed_batch_size}")

    for deterministic_flag in deterministic_flags:
        correct = 0
        total = 0
        print(f"\nTesting ONNX model with deterministic={deterministic_flag}")
        for batch in test_dataloader:
            images, labels = batch
            images = prepare_onnx_images(
                images.numpy().astype(np.float32),
                input_shape=input_shape,
            )
            labels = labels.numpy()

            if fixed_batch_size is not None and images.shape[0] != fixed_batch_size:
                print(
                    f"Skipping batch with size {images.shape[0]} "
                    f"(model expects {fixed_batch_size})"
                )
                continue

            inputs = {input_name: images}
            if deterministic_input is not None:
                inputs[deterministic_input] = np.array(deterministic_flag)

            try:
                preds = session.run([output_name], inputs)[0]
            except Exception as exc:
                print(f"ONNX Runtime error: {exc}")
                print(f"Input shape: {images.shape}, dtype: {images.dtype}")
                print(
                    "This likely means the model expects a different input shape or batch size."
                )
                print(
                    "Try setting --batch-size to the batch size used during ONNX export (e.g., 2)."
                )
                continue

            preds = np.argmax(preds, axis=1)
            correct += (preds == labels).sum()
            total += labels.shape[0]

        accuracy = correct / total if total > 0 else 0.0
        print(
            f"ONNX model test accuracy (deterministic={deterministic_flag}): {accuracy:.4f}"
        )
