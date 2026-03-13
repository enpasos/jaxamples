import numpy as np
import pytest

from jaxamples import mnist_onnx_eval


def test_prepare_onnx_images_keeps_nchw_layout():
    images = np.ones((2, 1, 28, 28), dtype=np.float32)

    converted = mnist_onnx_eval.prepare_onnx_images(
        images,
        input_shape=(2, 1, 28, 28),
    )

    assert converted.shape == (2, 1, 28, 28)
    assert np.array_equal(converted, images)


def test_prepare_onnx_images_transposes_to_nhwc():
    images = np.arange(2 * 1 * 3 * 4, dtype=np.float32).reshape(2, 1, 3, 4)

    converted = mnist_onnx_eval.prepare_onnx_images(
        images,
        input_shape=(2, 3, 4, 1),
    )

    assert converted.shape == (2, 3, 4, 1)
    assert np.array_equal(converted[..., 0], images[:, 0])


def test_prepare_onnx_images_rejects_unknown_layout():
    images = np.ones((2, 1, 28, 28), dtype=np.float32)

    with pytest.raises(ValueError, match="Unexpected ONNX input shape"):
        mnist_onnx_eval.prepare_onnx_images(
            images,
            input_shape=(2, 28, 28, 28),
        )
