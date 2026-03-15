import math
import os

import jax
import jax.numpy as jnp
import onnx
from flax import nnx
from jax2onnx import allclose, to_onnx

from jaxamples.mnist_config import (
    MnistCnnModelConfig,
    MnistExampleConfig,
    OnnxConfig,
    shared_mnist_training_config,
)
from jaxamples.mnist_example_cli import apply_common_overrides, parse_example_args
from jaxamples.mnist_example_runner import run_mnist_example
from jaxamples.mnist_training import (
    get_dataset_torch_dataloaders,
    load_model,
    resolve_checkpoint_resume,
    test_onnx_model,
    train_model,
    visualize_results,
)


def _pooled_size(size: int, num_pool_ops: int) -> int:
    pooled = size
    for _ in range(num_pool_ops):
        pooled = math.ceil(pooled / 2)
    return pooled


class MnistCnnClassifier(nnx.Module):
    """Strong convolutional baseline for MNIST under the shared augmentation setup."""

    def __init__(
        self,
        *,
        height: int,
        width: int,
        num_classes: int,
        conv_channels: list[int],
        dense_hidden_dim: int,
        feature_dropout_rate: float = 0.1,
        classifier_dropout_rate: float = 0.3,
        rngs: nnx.Rngs,
    ):
        params_key = rngs.params()
        (
            conv1_key,
            conv2_key,
            conv3_key,
            conv4_key,
            feature_dropout_key,
            hidden_key,
            classifier_dropout_key,
            out_key,
        ) = jax.random.split(params_key, 8)

        c1, c2, c3, c4 = conv_channels
        self.conv1 = nnx.Conv(
            in_features=1,
            out_features=c1,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=nnx.Rngs(conv1_key),
        )
        self.conv2 = nnx.Conv(
            in_features=c1,
            out_features=c2,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=nnx.Rngs(conv2_key),
        )
        self.conv3 = nnx.Conv(
            in_features=c2,
            out_features=c3,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=nnx.Rngs(conv3_key),
        )
        self.conv4 = nnx.Conv(
            in_features=c3,
            out_features=c4,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=nnx.Rngs(conv4_key),
        )
        self.feature_dropout = nnx.Dropout(
            rate=feature_dropout_rate,
            rngs=nnx.Rngs(feature_dropout_key),
        )

        pooled_height = _pooled_size(height, num_pool_ops=2)
        pooled_width = _pooled_size(width, num_pool_ops=2)
        flattened_dim = pooled_height * pooled_width * c4

        self.hidden = nnx.Linear(
            flattened_dim,
            dense_hidden_dim,
            rngs=nnx.Rngs(hidden_key),
        )
        self.classifier_dropout = nnx.Dropout(
            rate=classifier_dropout_rate,
            rngs=nnx.Rngs(classifier_dropout_key),
        )
        self.out = nnx.Linear(
            dense_hidden_dim,
            num_classes,
            rngs=nnx.Rngs(out_key),
        )

    def _feature_extractor(self, images: jax.Array, *, deterministic: bool) -> jax.Array:
        features = nnx.gelu(self.conv1(images), approximate=False)
        features = nnx.gelu(self.conv2(features), approximate=False)
        features = nnx.max_pool(
            features,
            window_shape=(2, 2),
            strides=(2, 2),
            padding="SAME",
        )
        features = self.feature_dropout(features, deterministic=deterministic)
        features = nnx.gelu(self.conv3(features), approximate=False)
        features = nnx.gelu(self.conv4(features), approximate=False)
        return nnx.max_pool(
            features,
            window_shape=(2, 2),
            strides=(2, 2),
            padding="SAME",
        )

    def __call__(self, images: jax.Array, *, deterministic: bool = True) -> jax.Array:
        features = self._feature_extractor(images, deterministic=deterministic)
        features = jnp.reshape(features, (features.shape[0], -1))
        features = nnx.gelu(self.hidden(features), approximate=False)
        features = self.classifier_dropout(features, deterministic=deterministic)
        return self.out(features)


def get_default_config() -> MnistExampleConfig:
    default_output_dir = os.path.abspath("./output/")
    model_config = MnistCnnModelConfig(
        height=28,
        width=28,
        num_classes=10,
        conv_channels=[32, 64, 128, 128],
        dense_hidden_dim=256,
        feature_dropout_rate=0.1,
        classifier_dropout_rate=0.3,
    )
    checkpoint_name = "cnn_c32_64_128_128_d256_checkpoints"
    return MnistExampleConfig(
        seed=5678,
        training=shared_mnist_training_config(
            checkpoint_dir=os.path.abspath(os.path.join("./data", checkpoint_name)),
            output_dir=default_output_dir,
        ),
        model=model_config,
        onnx=OnnxConfig(
            model_name="mnist_cnn_model",
            output_path=os.path.join(default_output_dir, "mnist_cnn_model.onnx"),
            input_shapes=[("B", 28, 28, 1)],
            input_params={"deterministic": True},
        ),
    )


def create_model(model_config: MnistCnnModelConfig, seed: int) -> MnistCnnClassifier:
    return MnistCnnClassifier(**model_config.to_dict(), rngs=nnx.Rngs(seed))


def main(args=None) -> None:
    cli_args = parse_example_args(
        args,
        description="Train and export the MNIST CNN baseline",
        default_onnx_output="output/mnist_cnn_model.onnx",
    )
    config = apply_common_overrides(get_default_config(), cli_args)
    run_mnist_example(
        config,
        create_model=create_model,
        get_dataloaders=get_dataset_torch_dataloaders,
        resolve_checkpoint_resume=resolve_checkpoint_resume,
        load_model=load_model,
        train_model=train_model,
        visualize_results=visualize_results,
        test_onnx_model=test_onnx_model,
        to_onnx_fn=to_onnx,
        allclose_fn=allclose,
        save_onnx_model=onnx.save_model,
    )


if __name__ == "__main__":
    main()
