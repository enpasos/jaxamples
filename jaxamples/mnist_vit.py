# file: jaxamples/mnist_vit.py
import os
from typing import Dict

import jax.numpy as jnp
import onnx
from flax import nnx
from jax2onnx import allclose, to_onnx
from jax2onnx.plugins.examples.nnx.vit import VisionTransformer

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
            checkpoint_dir=os.path.abspath("./data/checkpoints/"),
            output_dir=default_output_dir,
        ),
        model=MnistVitModelConfig(
            height=28,
            width=28,
            num_hiddens=256,
            num_layers=4,
            num_heads=4,
            mlp_dim=256,
            num_classes=10,
            embed_dims=[32, 128, 256],
            kernel_size=3,
            strides=[1, 2, 2],
            embedding_type="conv",
            embedding_dropout_rate=0.5,
            attention_dropout_rate=0.5,
            mlp_dropout_rate=0.5,
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
