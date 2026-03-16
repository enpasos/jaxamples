# file: jaxamples/mnist_vit.py
import os
from math import prod
from typing import Dict, Sequence

import jax.numpy as jnp
import onnx
from flax import nnx
from jax2onnx import allclose, to_onnx
from jax2onnx.plugins.examples.nnx.vit import (
    ConcatClsToken,
    ConvEmbedding,
    PatchEmbedding,
    PositionalEmbedding,
    TransformerStack,
)

from jaxamples import mnist_training as mnist_training_lib
from jaxamples.mnist_config import (
    MnistExampleConfig,
    MnistVitModelConfig,
    OnnxConfig,
    shared_mnist_training_config,
)
from jaxamples.mnist_example_cli import apply_common_overrides, parse_example_args
from jaxamples.mnist_example_runner import run_mnist_example


torchvision = mnist_training_lib.torchvision
get_dataset_torch_dataloaders = mnist_training_lib.get_dataset_torch_dataloaders
jax_collate = mnist_training_lib.jax_collate
rotate_image = mnist_training_lib.rotate_image
AugmentationParams = mnist_training_lib.AugmentationParams
augment_data_batch = mnist_training_lib.augment_data_batch
visualize_augmented_images = mnist_training_lib.visualize_augmented_images
loss_fn = mnist_training_lib.loss_fn
train_step = mnist_training_lib.train_step
eval_step = mnist_training_lib.eval_step
pred_step = mnist_training_lib.pred_step
visualize_incorrect_classifications = (
    mnist_training_lib.visualize_incorrect_classifications
)
lr_schedule = mnist_training_lib.lr_schedule
create_optimizer = mnist_training_lib.create_optimizer
compute_mean_and_spread = mnist_training_lib.compute_mean_and_spread
save_test_accuracy_metrics = mnist_training_lib.save_test_accuracy_metrics
load_and_plot_test_accuracy_metrics = (
    mnist_training_lib.load_and_plot_test_accuracy_metrics
)
save_and_plot_test_accuracy_metrics = (
    mnist_training_lib.save_and_plot_test_accuracy_metrics
)
save_model_visualization = mnist_training_lib.save_model_visualization
CKPT_EXTENSION = mnist_training_lib.CKPT_EXTENSION
save_model = mnist_training_lib.save_model
load_model = mnist_training_lib.load_model
resolve_checkpoint_resume = mnist_training_lib.resolve_checkpoint_resume
test_onnx_model = mnist_training_lib.test_onnx_model


def create_sinusoidal_embeddings(num_patches: int, num_hiddens: int) -> jnp.ndarray:
    position = jnp.arange(num_patches + 1)[:, jnp.newaxis]
    div_term = jnp.exp(
        jnp.arange(0, num_hiddens, 2) * -(jnp.log(10000.0) / num_hiddens)
    )
    pos_embedding = jnp.zeros((num_patches + 1, num_hiddens))
    pos_embedding = pos_embedding.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_embedding = pos_embedding.at[:, 1::2].set(jnp.cos(position * div_term))
    return pos_embedding[jnp.newaxis, :, :]


class VisionTransformer(nnx.Module):
    """MNIST-focused ViT with a stronger classifier head."""

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
        self.transformer_stack = TransformerStack(
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


def train_model(
    model: nnx.Module,
    start_epoch: int,
    metrics: nnx.MultiMetric,
    config: Dict,
    train_dataloader,
    test_dataloader,
    rng_key: jnp.ndarray,
):
    return mnist_training_lib.train_model(
        model,
        start_epoch,
        metrics,
        config,
        train_dataloader,
        test_dataloader,
        rng_key,
        augmentation_params_cls=AugmentationParams,
        create_optimizer_fn=create_optimizer,
        lr_schedule_fn=lr_schedule,
        jax_collate_fn=jax_collate,
        augment_data_batch_fn=augment_data_batch,
        train_step_fn=train_step,
        eval_step_fn=eval_step,
        visualize_augmented_images_fn=visualize_augmented_images,
        compute_mean_and_spread_fn=compute_mean_and_spread,
        save_test_accuracy_metrics_fn=save_test_accuracy_metrics,
        visualize_incorrect_classifications_fn=visualize_incorrect_classifications,
        save_model_fn=save_model,
        load_and_plot_test_accuracy_metrics_fn=load_and_plot_test_accuracy_metrics,
    )


def visualize_results(
    metrics_history: Dict[str, list],
    model: nnx.Module,
    test_dataloader,
    epoch: int,
    output_dir: str = "output",
):
    return mnist_training_lib.visualize_results(
        metrics_history,
        model,
        test_dataloader,
        epoch,
        output_dir=output_dir,
        jax_collate_fn=jax_collate,
        pred_step_fn=pred_step,
    )


def get_default_config() -> MnistExampleConfig:
    default_output_dir = os.path.abspath("./output/")
    default_onnx_dir = os.path.abspath("./onnx/")
    return MnistExampleConfig(
        seed=5678,
        training=shared_mnist_training_config(
            checkpoint_dir=os.path.abspath("./data/mnist_vit_cls_mean_checkpoints/"),
            output_dir=default_output_dir,
        ),
        model=MnistVitModelConfig(
            height=28,
            width=28,
            num_hiddens=256,
            num_layers=4,
            num_heads=4,
            mlp_dim=512,
            num_classes=10,
            embed_dims=[32, 128, 256],
            kernel_size=3,
            strides=[1, 2, 2],
            embedding_type="conv",
            embedding_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            mlp_dropout_rate=0.1,
            head_hidden_dim=256,
            head_dropout_rate=0.1,
            pool_features="cls_mean",
        ),
        onnx=OnnxConfig(
            model_name="mnist_vit_model",
            output_path=os.path.join(default_onnx_dir, "mnist_vit_model.onnx"),
            input_shapes=[("B", 28, 28, 1)],
            input_params={"deterministic": True},
        ),
    )


def create_model(model_config: MnistVitModelConfig, seed: int) -> nnx.Module:
    return VisionTransformer(**model_config.to_dict(), rngs=nnx.Rngs(seed))


def main(args=None) -> None:
    cli_args = parse_example_args(
        args,
        description="Train and export the MNIST ViT example",
        default_onnx_output="onnx/mnist_vit_model.onnx",
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
