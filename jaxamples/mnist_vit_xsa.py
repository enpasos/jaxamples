import os
from functools import wraps
from math import prod
from typing import Sequence

import jax
import jax.numpy as jnp
import onnx
from flax import nnx
from jax2onnx import allclose, to_onnx
from jax2onnx.plugins.examples.nnx.vit import (
    ConcatClsToken,
    ConvEmbedding,
    FeedForward,
    PatchEmbedding,
    PositionalEmbedding,
)
from jax2onnx.plugins.plugin_system import onnx_function

from jaxamples import mnist_vit as mnist_vit_lib
from jaxamples.mnist_config import MnistExampleConfig, MnistVitModelConfig, OnnxConfig
from jaxamples.mnist_example_cli import apply_common_overrides, parse_example_args
from jaxamples.mnist_example_runner import run_mnist_example


get_dataset_torch_dataloaders = mnist_vit_lib.get_dataset_torch_dataloaders
train_model = mnist_vit_lib.train_model
visualize_results = mnist_vit_lib.visualize_results
resolve_checkpoint_resume = mnist_vit_lib.resolve_checkpoint_resume
load_model = mnist_vit_lib.load_model
test_onnx_model = mnist_vit_lib.test_onnx_model


def _exclusive_self_attention(q, k, v, *args, **kwargs):
    y = nnx.dot_product_attention(q, k, v, *args, **kwargs)
    eps = jnp.asarray(1e-12, dtype=v.dtype)
    v_norm = v * jax.lax.rsqrt(jnp.sum(jnp.square(v), axis=-1, keepdims=True) + eps)
    return y - jnp.sum(y * v_norm, axis=-1, keepdims=True) * v_norm


@onnx_function(unique=True)
def exclusive_attention(*args, **kwargs):
    return _exclusive_self_attention(*args, **kwargs)


@wraps(exclusive_attention)
def _call_exclusive_attention(*args, **kwargs):
    return exclusive_attention(*args, **kwargs)


class ExclusiveMultiHeadAttention(nnx.Module):
    def __init__(
        self,
        num_hiddens: int,
        num_heads: int,
        attention_dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            qkv_features=num_hiddens,
            out_features=num_hiddens,
            in_features=num_hiddens,
            attention_fn=_call_exclusive_attention,
            rngs=rngs,
            decode=False,
        )
        self.dropout = nnx.Dropout(rate=attention_dropout_rate, rngs=rngs)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = self.attention(x, deterministic=deterministic)
        return self.dropout(x, deterministic=deterministic)


class ExclusiveTransformerBlock(nnx.Module):
    def __init__(
        self,
        num_hiddens: int,
        num_heads: int,
        mlp_dim: int,
        attention_dropout_rate: float = 0.1,
        mlp_dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.layer_norm1 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.attention = ExclusiveMultiHeadAttention(
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            attention_dropout_rate=attention_dropout_rate,
            rngs=rngs,
        )
        self.layer_norm2 = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.mlp_block = FeedForward(
            num_hiddens,
            mlp_dim,
            mlp_dropout_rate,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        r = self.layer_norm1(x)
        r = self.attention(r, deterministic=deterministic)
        x = x + r
        r = self.layer_norm2(x)
        return x + self.mlp_block(r, deterministic=deterministic)


class ExclusiveTransformerStack(nnx.Module):
    def __init__(
        self,
        num_hiddens: int,
        num_heads: int,
        mlp_dim: int,
        num_layers: int,
        attention_dropout_rate: float = 0.1,
        mlp_dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        self.blocks = nnx.List(
            [
                ExclusiveTransformerBlock(
                    num_hiddens,
                    num_heads,
                    mlp_dim,
                    attention_dropout_rate,
                    mlp_dropout_rate,
                    rngs=rngs,
                )
                for _ in range(num_layers)
            ]
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for block in self.blocks:
            x = block(x, deterministic=deterministic)
        return x


class VisionTransformer(nnx.Module):
    """MNIST-focused ViT with an exclusive self-attention stack."""

    def __init__(
        self,
        height: int,
        width: int,
        num_hiddens: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        num_classes: int,
        embed_dims: Sequence[int] = (32, 128, 256),
        kernel_size: int = 3,
        strides: Sequence[int] = (1, 2, 2),
        embedding_type: str = "conv",
        embedding_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        mlp_dropout_rate: float = 0.1,
        patch_size: int = 4,
        head_hidden_dim: int = 256,
        head_dropout_rate: float = 0.1,
        pool_features: str = "cls_mean",
        *,
        rngs: nnx.Rngs,
    ):
        if embedding_type not in {"conv", "patch"}:
            raise ValueError("embedding_type must be either 'conv' or 'patch'")
        if pool_features not in {"cls", "cls_mean"}:
            raise ValueError("pool_features must be either 'cls' or 'cls_mean'")

        self.pool_features = pool_features
        embed_dims = tuple(embed_dims)
        strides = tuple(strides)

        if embedding_type == "conv":
            self.embedding = ConvEmbedding(
                W=width,
                H=height,
                embed_dims=embed_dims,
                kernel_size=kernel_size,
                strides=strides,
                dropout_rate=embedding_dropout_rate,
                rngs=rngs,
            )
            downsample_factor = prod(strides)
            num_patches = (height // downsample_factor) * (width // downsample_factor)
        else:
            self.embedding = PatchEmbedding(
                height=height,
                width=width,
                patch_size=patch_size,
                num_hiddens=num_hiddens,
                in_features=1,
                rngs=rngs,
            )
            num_patches = (height // patch_size) * (width // patch_size)

        self.concat_cls_token = ConcatClsToken(num_hiddens=num_hiddens, rngs=rngs)
        self.positional_embedding = PositionalEmbedding(
            num_hiddens=num_hiddens,
            num_patches=num_patches,
        )
        self.transformer_stack = ExclusiveTransformerStack(
            num_hiddens,
            num_heads,
            mlp_dim,
            num_layers,
            attention_dropout_rate,
            mlp_dropout_rate,
            rngs=rngs,
        )

        head_input_dim = num_hiddens if pool_features == "cls" else num_hiddens * 2
        self.head_norm = nnx.LayerNorm(head_input_dim, rngs=rngs)
        self.head_hidden = nnx.Linear(head_input_dim, head_hidden_dim, rngs=rngs)
        self.head_dropout = nnx.Dropout(rate=head_dropout_rate, rngs=rngs)
        self.head_out = nnx.Linear(head_hidden_dim, num_classes, rngs=rngs)

    def _pool_tokens(self, tokens: jnp.ndarray) -> jnp.ndarray:
        cls_token = tokens[:, 0, :]
        if self.pool_features == "cls":
            return cls_token

        patch_tokens = tokens[:, 1:, :]
        mean_pooled = jnp.mean(patch_tokens, axis=1)
        return jnp.concatenate([cls_token, mean_pooled], axis=-1)

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        if x is None or x.shape[0] == 0:
            raise ValueError("Input tensor 'x' must not be None or empty.")
        if len(x.shape) != 4 or x.shape[-1] != 1:
            raise ValueError("Input tensor 'x' must have shape (B, H, W, 1).")

        x = self.embedding(x, deterministic=deterministic)
        x = self.concat_cls_token(x)
        x = self.positional_embedding(x)
        x = self.transformer_stack(x, deterministic=deterministic)
        x = self._pool_tokens(x)
        x = self.head_norm(x)
        x = self.head_hidden(x)
        x = nnx.gelu(x, approximate=False)
        x = self.head_dropout(x, deterministic=deterministic)
        return self.head_out(x)


def get_default_config() -> MnistExampleConfig:
    base_config = mnist_vit_lib.get_default_config()
    default_output_dir = os.path.abspath("./output/mnist_vit_xsa/")
    default_onnx_dir = os.path.abspath("./onnx/")
    base_config.training.checkpoint_dir = os.path.abspath(
        "./data/mnist_vit_xsa_cls_mean_checkpoints/"
    )
    base_config.training.output_dir = default_output_dir
    return MnistExampleConfig(
        seed=base_config.seed,
        training=base_config.training,
        model=base_config.model,
        onnx=OnnxConfig(
            model_name="mnist_vit_xsa_model",
            output_path=os.path.join(default_onnx_dir, "mnist_vit_xsa_model.onnx"),
            input_shapes=base_config.onnx.input_shapes,
            input_params=base_config.onnx.input_params,
        ),
    )


def create_model(model_config: MnistVitModelConfig, seed: int) -> nnx.Module:
    return VisionTransformer(**model_config.to_dict(), rngs=nnx.Rngs(seed))


def main(args=None) -> None:
    cli_args = parse_example_args(
        args,
        description="Train and export the MNIST ViT XSA example",
        default_onnx_output="onnx/mnist_vit_xsa_model.onnx",
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
