import math
import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from flax import nnx
from jax2onnx import allclose, to_onnx
from jax2onnx.plugins.examples.nnx.dinov3 import (
    DinoRotaryProcessHeads,
    VisionTransformer as DinoVisionTransformer,
)

from jaxamples.mnist_config import (
    MnistDinoV3ModelConfig,
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


def prepare_dinov3_inputs(
    images: jax.Array,
    *,
    expected_size: Optional[int] = None,
) -> jax.Array:
    """Convert MNIST NHWC inputs into the NCHW 3-channel layout DINOv3 expects."""
    if images.ndim != 4:
        raise ValueError(f"Expected 4D image batch, got shape {images.shape}.")

    batch_size, height, width, channels = images.shape
    if expected_size is not None and (height != expected_size or width != expected_size):
        raise ValueError(
            f"Expected {expected_size}x{expected_size} images, got {height}x{width}."
        )
    if channels not in (1, 3):
        raise ValueError(f"Expected 1 or 3 channels, got {channels}.")

    nchw_images = jnp.transpose(images, (0, 3, 1, 2))
    if channels == 1:
        nchw_images = jnp.repeat(nchw_images, repeats=3, axis=1)

    assert nchw_images.shape[0] == batch_size
    return nchw_images


class MnistDinoV3Classifier(nnx.Module):
    """Small MNIST classifier using the DINOv3 ViT backbone from jax2onnx."""

    def __init__(
        self,
        *,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        num_classes: int,
        num_storage_tokens: int = 0,
        head_hidden_dim: int = 192,
        head_dropout_rate: float = 0.1,
        pool_features: str = "cls_mean",
        rngs: nnx.Rngs,
    ):
        params_key = rngs.params()
        backbone_key, head_key = jax.random.split(params_key)
        head_norm_key, head_hidden_key, head_dropout_key, head_out_key = jax.random.split(
            head_key, 4
        )

        self.img_size = int(img_size)
        self.pool_features = pool_features
        self.backbone = DinoVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_storage_tokens=num_storage_tokens,
            rngs=nnx.Rngs(backbone_key),
        )
        grid_size = math.isqrt(self.backbone.patch_embed.num_patches)
        if grid_size * grid_size != self.backbone.patch_embed.num_patches:
            raise ValueError("DINOv3 patch grid must be square for rotary embeddings.")
        sin, cos = self.backbone.dino_rope.get_sincos(
            H=grid_size,
            W=grid_size,
            inference=True,
        )
        self.process_heads = DinoRotaryProcessHeads(
            sin=np.asarray(sin),
            cos=np.asarray(cos),
            prefix_tokens=1 + num_storage_tokens,
        )
        head_input_dim = embed_dim if pool_features == "cls" else embed_dim * 2
        self.head_norm = nnx.LayerNorm(head_input_dim, rngs=nnx.Rngs(head_norm_key))
        self.head_hidden = nnx.Linear(
            head_input_dim, head_hidden_dim, rngs=nnx.Rngs(head_hidden_key)
        )
        self.head_dropout = nnx.Dropout(rate=head_dropout_rate, rngs=nnx.Rngs(head_dropout_key))
        self.head_out = nnx.Linear(
            head_hidden_dim, num_classes, rngs=nnx.Rngs(head_out_key)
        )

    def _encode_backbone(self, images: jax.Array) -> jax.Array:
        tokens = self.backbone.patch_embed(images)
        cls_tokens = jnp.broadcast_to(
            self.backbone.cls_token[...], (tokens.shape[0], 1, tokens.shape[-1])
        )
        if self.backbone.num_storage_tokens and self.backbone.storage_tokens is not None:
            storage_tokens = jnp.broadcast_to(
                self.backbone.storage_tokens[...],
                (tokens.shape[0], self.backbone.num_storage_tokens, tokens.shape[-1]),
            )
            tokens = jnp.concatenate([cls_tokens, storage_tokens, tokens], axis=1)
        else:
            tokens = jnp.concatenate([cls_tokens, tokens], axis=1)

        for block in self.backbone.blocks:
            tokens = block(tokens, process_heads=self.process_heads)
        return self.backbone.norm(tokens)

    def _pool_head_features(self, tokens: jax.Array) -> jax.Array:
        cls_token = tokens[:, 0, :]
        if self.pool_features == "cls":
            return cls_token

        patch_tokens = tokens[:, 1 + self.backbone.num_storage_tokens :, :]
        mean_pooled = jnp.mean(patch_tokens, axis=1)
        return jnp.concatenate([cls_token, mean_pooled], axis=-1)

    def __call__(
        self, images: jax.Array, *, deterministic: bool = True
    ) -> jax.Array:
        backbone_inputs = prepare_dinov3_inputs(images, expected_size=self.img_size)
        tokens = self._encode_backbone(backbone_inputs)
        head_features = self._pool_head_features(tokens)
        head_features = self.head_norm(head_features)
        head_features = self.head_hidden(head_features)
        head_features = nnx.gelu(head_features, approximate=False)
        head_features = self.head_dropout(head_features, deterministic=deterministic)
        return self.head_out(head_features)


def get_default_config() -> MnistExampleConfig:
    default_output_dir = os.path.abspath("./output/")
    default_onnx_dir = os.path.abspath("./onnx/")
    model_config = MnistDinoV3ModelConfig(
        img_size=28,
        patch_size=4,
        embed_dim=192,
        depth=4,
        num_heads=6,
        num_classes=10,
        num_storage_tokens=0,
        head_hidden_dim=192,
        head_dropout_rate=0.1,
        pool_features="cls_mean",
    )
    checkpoint_name = (
        "dinov3_"
        f"p{model_config.patch_size}_"
        f"dim{model_config.embed_dim}_"
        f"d{model_config.depth}_"
        f"h{model_config.num_heads}_"
        f"{model_config.pool_features}_checkpoints"
    )
    return MnistExampleConfig(
        seed=5678,
        training=shared_mnist_training_config(
            checkpoint_dir=os.path.abspath(os.path.join("./data", checkpoint_name)),
            output_dir=default_output_dir,
        ),
        # Match the ViT example more closely on token count and parameter budget.
        model=model_config,
        onnx=OnnxConfig(
            model_name="mnist_dinov3_model",
            output_path=os.path.join(default_onnx_dir, "mnist_dinov3_model.onnx"),
            input_shapes=[("B", 28, 28, 1)],
            input_params={"deterministic": True},
        ),
    )


def create_model(
    model_config: MnistDinoV3ModelConfig, seed: int
) -> MnistDinoV3Classifier:
    return MnistDinoV3Classifier(**model_config.to_dict(), rngs=nnx.Rngs(seed))


def main(args=None) -> None:
    cli_args = parse_example_args(
        args,
        description="Train and export the MNIST DINOv3 example",
        default_onnx_output="onnx/mnist_dinov3_model.onnx",
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
