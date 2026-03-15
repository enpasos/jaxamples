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


class ResidualConvBlock(nnx.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        strides: tuple[int, int] = (1, 1),
        rngs: nnx.Rngs,
    ):
        params_key = rngs.params()
        conv1_key, norm1_key, conv2_key, norm2_key, proj_key, proj_norm_key = jax.random.split(
            params_key, 6
        )

        self.conv1 = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(3, 3),
            strides=strides,
            padding="SAME",
            use_bias=False,
            rngs=nnx.Rngs(conv1_key),
        )
        self.norm1 = nnx.LayerNorm(out_features, rngs=nnx.Rngs(norm1_key))
        self.conv2 = nnx.Conv(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            rngs=nnx.Rngs(conv2_key),
        )
        self.norm2 = nnx.LayerNorm(out_features, rngs=nnx.Rngs(norm2_key))

        if in_features != out_features or strides != (1, 1):
            self.proj = nnx.Conv(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(1, 1),
                strides=strides,
                padding="SAME",
                use_bias=False,
                rngs=nnx.Rngs(proj_key),
            )
            self.proj_norm = nnx.LayerNorm(
                out_features,
                rngs=nnx.Rngs(proj_norm_key),
            )
        else:
            self.proj = None
            self.proj_norm = None

    def __call__(self, inputs: jax.Array) -> jax.Array:
        residual = inputs
        outputs = self.conv1(inputs)
        outputs = self.norm1(outputs)
        outputs = nnx.gelu(outputs, approximate=False)
        outputs = self.conv2(outputs)
        outputs = self.norm2(outputs)

        if self.proj is not None and self.proj_norm is not None:
            residual = self.proj(residual)
            residual = self.proj_norm(residual)

        return nnx.gelu(outputs + residual, approximate=False)


def _pooled_size(size: int, num_stride_two_ops: int) -> int:
    pooled = size
    for _ in range(num_stride_two_ops):
        pooled = math.ceil(pooled / 2)
    return pooled


class MnistStrongCnnClassifier(nnx.Module):
    """Residual CNN with LayerNorm and a spatially preserving classifier head."""

    def __init__(
        self,
        *,
        height: int,
        width: int,
        num_classes: int,
        conv_channels: list[int],
        dense_hidden_dim: int,
        feature_dropout_rate: float = 0.05,
        classifier_dropout_rate: float = 0.15,
        rngs: nnx.Rngs,
    ):
        params_key = rngs.params()
        (
            stem_key,
            stem_norm_key,
            block1_key,
            block2_key,
            block3_key,
            feature_dropout_key,
            hidden_key,
            classifier_dropout_key,
            out_key,
        ) = jax.random.split(params_key, 9)

        c1, c2, c3, c4 = conv_channels
        self.stem = nnx.Conv(
            in_features=1,
            out_features=c1,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            rngs=nnx.Rngs(stem_key),
        )
        self.stem_norm = nnx.LayerNorm(c1, rngs=nnx.Rngs(stem_norm_key))
        self.block1 = ResidualConvBlock(
            in_features=c1,
            out_features=c2,
            strides=(2, 2),
            rngs=nnx.Rngs(block1_key),
        )
        self.block2 = ResidualConvBlock(
            in_features=c2,
            out_features=c3,
            strides=(2, 2),
            rngs=nnx.Rngs(block2_key),
        )
        self.block3 = ResidualConvBlock(
            in_features=c3,
            out_features=c4,
            strides=(1, 1),
            rngs=nnx.Rngs(block3_key),
        )
        self.feature_dropout = nnx.Dropout(
            rate=feature_dropout_rate,
            rngs=nnx.Rngs(feature_dropout_key),
        )

        pooled_height = _pooled_size(height, num_stride_two_ops=2)
        pooled_width = _pooled_size(width, num_stride_two_ops=2)
        flattened_dim = pooled_height * pooled_width * c4

        self.hidden = nnx.Linear(
            flattened_dim, dense_hidden_dim, rngs=nnx.Rngs(hidden_key)
        )
        self.classifier_dropout = nnx.Dropout(
            rate=classifier_dropout_rate,
            rngs=nnx.Rngs(classifier_dropout_key),
        )
        self.out = nnx.Linear(dense_hidden_dim, num_classes, rngs=nnx.Rngs(out_key))

    def __call__(self, images: jax.Array, *, deterministic: bool = True) -> jax.Array:
        features = self.stem(images)
        features = self.stem_norm(features)
        features = nnx.gelu(features, approximate=False)
        features = self.block1(features)
        features = self.block2(features)
        features = self.block3(features)
        features = self.feature_dropout(features, deterministic=deterministic)
        features = jnp.reshape(features, (features.shape[0], -1))
        features = nnx.gelu(self.hidden(features), approximate=False)
        features = self.classifier_dropout(features, deterministic=deterministic)
        return self.out(features)


def get_default_config() -> MnistExampleConfig:
    default_output_dir = os.path.abspath("./output/")
    default_onnx_dir = os.path.abspath("./onnx/")
    model_config = MnistCnnModelConfig(
        height=28,
        width=28,
        num_classes=10,
        conv_channels=[32, 64, 128, 192],
        dense_hidden_dim=256,
        feature_dropout_rate=0.05,
        classifier_dropout_rate=0.15,
    )
    checkpoint_name = "strong_cnn_residual_ln_checkpoints"
    return MnistExampleConfig(
        seed=5678,
        training=shared_mnist_training_config(
            checkpoint_dir=os.path.abspath(os.path.join("./data", checkpoint_name)),
            output_dir=default_output_dir,
        ),
        model=model_config,
        onnx=OnnxConfig(
            model_name="mnist_strong_cnn_model",
            output_path=os.path.join(default_onnx_dir, "mnist_strong_cnn_model.onnx"),
            input_shapes=[("B", 28, 28, 1)],
            input_params={"deterministic": True},
        ),
    )


def create_model(
    model_config: MnistCnnModelConfig, seed: int
) -> MnistStrongCnnClassifier:
    return MnistStrongCnnClassifier(**model_config.to_dict(), rngs=nnx.Rngs(seed))


def main(args=None) -> None:
    cli_args = parse_example_args(
        args,
        description="Train and export the strong MNIST CNN baseline",
        default_onnx_output="onnx/mnist_strong_cnn_model.onnx",
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
