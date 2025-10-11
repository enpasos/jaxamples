# file: jaxamples/mnist_vit.py
import functools
import os

# Suppress XLA compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import shutil
import zipfile
import warnings
from typing import Dict, Tuple, List, Any
from flax.struct import dataclass, field

import jax.lax
import math
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.ndimage import map_coordinates
from jax import Array
import onnx
import optax
import torchvision
import treescope
from flax import nnx
from jax.image import scale_and_translate

from torch.utils.data import DataLoader
from torchvision import transforms
from jax2onnx import to_onnx, allclose
from jax2onnx.plugins.examples.nnx.vit import VisionTransformer

# from jaxamples.vit import VisionTransformer
import orbax.checkpoint as orbax

import matplotlib
import numpy as np
import onnxruntime as ort

matplotlib.use("Agg")  # Use a non-interactive backend to avoid Tkinter-related issues
import matplotlib.pyplot as plt
import csv

warnings.filterwarnings(
    "ignore", message="Couldn't find sharding info under RestoreArgs.*"
)


# =============================================================================
# Data, augmentation and model utility functions
# =============================================================================


def get_dataset_torch_dataloaders(batch_size: int, data_dir: str = "./data"):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = torchvision.datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    train_dataloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    test_dataloader = DataLoader(
        test_ds, batch_size=1000, shuffle=False, num_workers=0, drop_last=True
    )
    return train_dataloader, test_dataloader


def jax_collate(batch):
    images, labels = batch
    images = jnp.array(images.numpy())
    labels = jnp.array(labels.numpy())
    images = jnp.transpose(images, (0, 2, 3, 1))
    return {"image": images, "label": labels}


def rotate_image(image: jnp.ndarray, angle: float) -> jnp.ndarray:
    """Rotates an image."""
    angle_rad = jnp.deg2rad(angle)
    cos_angle = jnp.cos(angle_rad)
    sin_angle = jnp.sin(angle_rad)
    center_y, center_x = jnp.array(image.shape[:2]) / 2.0
    yy, xx = jnp.meshgrid(
        jnp.arange(image.shape[0]), jnp.arange(image.shape[1]), indexing="ij"
    )
    yy = yy - center_y
    xx = xx - center_x
    rotated_x = cos_angle * xx + sin_angle * yy + center_x
    rotated_y = -sin_angle * xx + cos_angle * yy + center_y
    rotated_image = map_coordinates(
        image[..., 0], [rotated_y.ravel(), rotated_x.ravel()], order=1, mode="constant"
    )
    rotated_image = rotated_image.reshape(image.shape[:2])
    return jnp.expand_dims(rotated_image, axis=-1)


def jax_gaussian_filter(x: jnp.ndarray, sigma: float, radius: int) -> jnp.ndarray:
    """Performs 2D Gaussian filtering on array x using separable convolutions."""
    size = 2 * radius + 1
    ax = jnp.arange(-radius, radius + 1, dtype=x.dtype)
    kernel = jnp.exp(-0.5 * (ax / sigma) ** 2)
    kernel = kernel / jnp.sum(kernel)
    kernel_h = kernel.reshape((size, 1, 1, 1))
    kernel_v = kernel.reshape((1, size, 1, 1))
    x_exp = x[None, ..., None]
    x_filtered = jax.lax.conv_general_dilated(
        x_exp,
        kernel_h,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    x_filtered = jax.lax.conv_general_dilated(
        x_filtered,
        kernel_v,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return jnp.squeeze(x_filtered, axis=(0, 3))


@functools.partial(jax.jit, static_argnames=("radius",))
def elastic_deform(
    image: jnp.ndarray,
    alpha: float,
    sigma: float,
    rng_key: jnp.ndarray,
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    radius: int,
) -> jnp.ndarray:
    shape = image.shape[:2]
    key_dx, key_dy = random.split(rng_key, 2)
    dx = random.normal(key_dx, shape) * alpha
    dy = random.normal(key_dy, shape) * alpha

    dx = jax_gaussian_filter(dx, sigma, radius)
    dy = jax_gaussian_filter(dy, sigma, radius)

    indices = (jnp.reshape(y_grid + dy, (-1, 1)), jnp.reshape(x_grid + dx, (-1, 1)))
    deformed_image = map_coordinates(image[..., 0], indices, order=1, mode="reflect")
    return jnp.expand_dims(deformed_image.reshape(shape), axis=-1)


def visualize_augmented_images(
    ds: Dict[str, jnp.ndarray], epoch: int, num_images: int = 9
) -> None:
    """
    Displays a grid of augmented images.

    Args:
        ds (Dict[str, jnp.ndarray]): The dataset containing the augmented images.
        num_images (int, optional): The number of images to display. Defaults to 9.
    """

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(ds["image"][i, ..., 0], cmap="gray")
        ax.axis("off")

    plt.savefig(f"output/augmented_images_epoch{epoch}.png")
    plt.close(fig)


@dataclass
class AugmentationParams:
    # dynamic hyperparameters
    max_translation: float
    scale_min_x:    float
    scale_max_x:    float
    scale_min_y:    float
    scale_max_y:    float
    max_rotation:   float
    elastic_alpha:  float
    elastic_sigma:  float

    # static switches — won't cause retrace when changed
    enable_elastic:      bool = field(pytree_node=False)
    enable_rotation:     bool = field(pytree_node=False)
    enable_scaling:      bool = field(pytree_node=False)
    enable_translation:  bool = field(pytree_node=False)
    enable_rect_erasing: bool = field(pytree_node=False)
    rect_erase_height:   int  = field(pytree_node=False)
    rect_erase_width:    int  = field(pytree_node=False)


@functools.partial(jax.jit, static_argnames=("augmentation_params",))
def augment_data_batch(
    batch: Dict[str, jnp.ndarray],
    rng_key: jnp.ndarray,
    augmentation_params: AugmentationParams,
) -> Dict[str, jnp.ndarray]:
    images = batch["image"]
    batch_size, height, width, channels = images.shape

    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    x_grid = jnp.array(x_grid)
    y_grid = jnp.array(y_grid)

    alpha_val = augmentation_params.elastic_alpha
    sigma_val = augmentation_params.elastic_sigma
    radius_val = math.ceil(3.0 * sigma_val)

    def augment_single_image(image, key):
        key1, key2, key3, key4, key5, key6, key7, key8 = random.split(key, 8)
        max_translation = augmentation_params.max_translation
        tx = random.uniform(key1, minval=-max_translation, maxval=max_translation)
        ty = random.uniform(key2, minval=-max_translation, maxval=max_translation)
        translation = jnp.array([ty, tx])
        scale_factor_x = random.uniform(
            key3,
            minval=augmentation_params.scale_min_x,
            maxval=augmentation_params.scale_max_x,
        )
        scale_factor_y = random.uniform(
            key4,
            minval=augmentation_params.scale_min_y,
            maxval=augmentation_params.scale_max_y,
        )
        scale = jnp.array([scale_factor_y, scale_factor_x])
        max_rotation = augmentation_params.max_rotation * (jnp.pi / 180.0)
        rotation_angle = random.uniform(key5, minval=-max_rotation, maxval=max_rotation)

        if augmentation_params.enable_elastic:
            image = elastic_deform(
                image, alpha_val, sigma_val, key6, x_grid, y_grid, radius_val
            )

        if augmentation_params.enable_rotation:
            image = rotate_image(image, jnp.rad2deg(rotation_angle))

        if augmentation_params.enable_scaling or augmentation_params.enable_translation:
            image = scale_and_translate(
                image=image,
                shape=(height, width, channels),
                spatial_dims=(0, 1),
                scale=(
                    scale
                    if augmentation_params.enable_scaling
                    else jnp.array([1.0, 1.0])
                ),
                translation=(
                    translation
                    if augmentation_params.enable_translation
                    else jnp.array([0.0, 0.0])
                ),
                method="linear",
                antialias=True,
            )

        # Rectangle erasing augmentation
        if getattr(augmentation_params, "enable_rect_erasing", False):
            erase_h = getattr(augmentation_params, "rect_erase_height", 6)
            erase_w = getattr(augmentation_params, "rect_erase_width", 6)
            # Random top-left corner
            max_y = height - erase_h
            max_x = width - erase_w
            y0 = jax.lax.floor(
                random.uniform(key7, (), minval=0, maxval=max_y + 1)
            ).astype(jnp.int32)
            x0 = jax.lax.floor(
                random.uniform(key8, (), minval=0, maxval=max_x + 1)
            ).astype(jnp.int32)
            # Erase (set to 0) using dynamic_update_slice
            mask = jnp.ones_like(image)
            erase_shape = (erase_h, erase_w, channels)
            zero_patch = jnp.zeros(erase_shape, dtype=image.dtype)
            mask = jax.lax.dynamic_update_slice(mask, zero_patch, (y0, x0, 0))
            image = image * mask
        return image

    rng_keys = random.split(rng_key, num=batch_size)
    augmented_images = jax.vmap(augment_single_image)(images, rng_keys)
    return {"image": augmented_images, "label": batch["label"]}


def create_sinusoidal_embeddings(num_patches: int, num_hiddens: int) -> jnp.ndarray:
    position = jnp.arange(num_patches + 1)[:, jnp.newaxis]
    div_term = jnp.exp(
        jnp.arange(0, num_hiddens, 2) * -(jnp.log(10000.0) / num_hiddens)
    )
    pos_embedding = jnp.zeros((num_patches + 1, num_hiddens))
    pos_embedding = pos_embedding.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_embedding = pos_embedding.at[:, 1::2].set(jnp.cos(position * div_term))
    return pos_embedding[jnp.newaxis, :, :]


def loss_fn(
    model: nnx.Module, batch: Dict[str, jnp.ndarray], deterministic: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    logits = model(batch["image"], deterministic=deterministic)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    return loss, logits


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch: Dict[str, jnp.ndarray],
    learning_rate: float,
    weight_decay: float,
):
    model.train()
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])
    # Update hyperparameters before calling update
    optimizer.opt_state.hyperparams['learning_rate'] = learning_rate
    optimizer.opt_state.hyperparams['weight_decay'] = weight_decay
    optimizer.update(model, grads)


@nnx.jit
def eval_step(
    model: nnx.Module, metrics: nnx.MultiMetric, batch: Dict[str, jnp.ndarray]
):
    model.eval()
    loss, logits = loss_fn(model, batch, deterministic=True)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])


@nnx.jit
def pred_step(model: nnx.Module, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    model.eval()
    logits = model(batch["image"], deterministic=True)
    return jnp.argmax(logits, axis=1)


def visualize_incorrect_classifications(
    model: nnx.Module,
    test_dataloader: DataLoader,
    epoch: int,
    figsize: Tuple[int, int] = (15, 5),
) -> None:
    """
    Displays a grid of incorrectly classified images.

    If there are more than 50 misclassified images, the visualization is skipped.

    Args:
        model: The trained model.
        test_dataloader: DataLoader for the test dataset.
        epoch: Current epoch (used for the output filename).
        figsize: Figure size for matplotlib.
    """
    incorrect_images = []
    incorrect_labels = []
    incorrect_preds = []

    for batch in test_dataloader:
        batch = jax_collate(batch)
        preds = pred_step(model, batch)
        incorrect_mask = preds != batch["label"]
        if incorrect_mask.any():
            incorrect_images.append(batch["image"][incorrect_mask])
            incorrect_labels.append(batch["label"][incorrect_mask])
            incorrect_preds.append(preds[incorrect_mask])

    # Concatenate all incorrect data
    if incorrect_images:
        incorrect_images = jnp.concatenate(incorrect_images, axis=0)
        incorrect_images_typed: Array = incorrect_images  # to make mypy happy
        incorrect_labels = jnp.concatenate(incorrect_labels, axis=0)
        incorrect_preds = jnp.concatenate(incorrect_preds, axis=0)
    else:
        print("No incorrect classifications found.")
        return

    num_images = len(incorrect_images)

    if num_images > 50:
        print(
            f"Too many incorrect classifications ({num_images}). Skipping visualization."
        )
        return

    # If 50 or fewer, display all
    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    if num_images == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot
    for i, ax in enumerate(axes):
        ax.imshow(incorrect_images_typed[i, ..., 0], cmap="gray")
        ax.set_title(f"{incorrect_labels[i]}\nbut\n{incorrect_preds[i]}")
        ax.axis("off")

    plt.savefig(f"output/incorrect_classifications_epoch{epoch}.png")
    plt.close(fig)


# =============================================================================
# Learning rate scheduling and optimizer creation
# =============================================================================


def lr_schedule(epoch: int, config: Dict) -> float:
    total_epochs = (
        config["training"]["start_epoch"]
        + config["training"]["num_epochs_to_train_now"]
    )
    base_lr = config["training"]["base_learning_rate"]
    # Cosine schedule computed over the full training duration
    return 0.5 * base_lr * (1 + jnp.cos(jnp.pi * epoch / total_epochs))


def create_optimizer(
    model: nnx.Module, learning_rate: float, weight_decay: float
) -> nnx.Optimizer:
    # Use inject_hyperparams to allow dynamic learning rate changes
    tx = optax.inject_hyperparams(optax.adamw)(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


# =============================================================================
# Training, evaluation, checkpointing and visualization functions
# =============================================================================


def compute_mean_and_spread(values: List[float]) -> Tuple[float, float]:
    """Compute the mean and spread (standard deviation) of a list of values."""
    mean = np.mean(values)
    spread = np.std(values)
    return mean, spread


def save_test_accuracy_metrics(
    metrics_history: Dict[str, List[float]], epoch: int
) -> None:
    """Saves test accuracy, mean, and spread metrics to a CSV file."""
    output_csv = "output/test_accuracy_metrics.csv"
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(
                [
                    "epoch",
                    "train_accuracy",
                    "train_mean_accuracy",
                    "train_spread_accuracy",
                    "test_accuracy",
                    "test_mean_accuracy",
                    "test_spread_accuracy",
                ]
            )
        writer.writerow(
            [
                epoch,
                metrics_history["train_accuracy"][-1],
                (
                    metrics_history["train_accuracy_mean"][-1]
                    if metrics_history["train_accuracy_mean"][-1] is not None
                    else "N/A"
                ),
                (
                    metrics_history["train_accuracy_spread"][-1]
                    if metrics_history["train_accuracy_spread"][-1] is not None
                    else "N/A"
                ),
                metrics_history["test_accuracy"][-1],
                (
                    metrics_history["test_accuracy_mean"][-1]
                    if metrics_history["test_accuracy_mean"][-1] is not None
                    else "N/A"
                ),
                (
                    metrics_history["test_accuracy_spread"][-1]
                    if metrics_history["test_accuracy_spread"][-1] is not None
                    else "N/A"
                ),
            ]
        )
    print(f"Test and train accuracy metrics for epoch {epoch} saved to {output_csv}")


def load_and_plot_test_accuracy_metrics(csv_filepath: str, output_fig: str) -> None:
    """Loads test accuracy metrics from a CSV file and generates a plot."""
    epochs = []
    train_acc = []
    train_mean = []
    train_spread = []
    test_acc = []
    test_mean = []
    test_spread = []
    with open(csv_filepath, mode="r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            # Only plot if both means are not zero (skip warmup rows)
            if (
                float(row["test_mean_accuracy"]) != 0.0
                and float(row["train_mean_accuracy"]) != 0.0
            ):
                epochs.append(int(row["epoch"]))
                train_acc.append(float(row["train_accuracy"]))
                train_mean.append(float(row["train_mean_accuracy"]))
                train_spread.append(float(row["train_spread_accuracy"]))
                test_acc.append(float(row["test_accuracy"]))
                test_mean.append(float(row["test_mean_accuracy"]))
                test_spread.append(float(row["test_spread_accuracy"]))
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, test_acc, label="Test Accuracy", marker="o", linestyle="-")
    ax.plot(epochs, test_mean, label="Test Mean (last 10)", marker="s", linestyle="--")
    ax.fill_between(
        epochs,
        np.array(test_mean) - np.array(test_spread),
        np.array(test_mean) + np.array(test_spread),
        color="gray",
        alpha=0.2,
        label="Test Spread (±1 std)",
    )
    ax.plot(epochs, train_acc, label="Train Accuracy", marker="^", linestyle="-")
    ax.plot(epochs, train_mean, label="Train Mean (last 10)", marker="x", linestyle=":")
    ax.fill_between(
        epochs,
        np.array(train_mean) - np.array(train_spread),
        np.array(train_mean) + np.array(train_spread),
        color="blue",
        alpha=0.1,
        label="Train Spread (±1 std)",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train & Test Accuracy Metrics Over Epochs")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(output_fig, dpi=300)
    plt.close(fig)
    print(f"Test and train accuracy graph saved to {output_fig}")


def save_and_plot_test_accuracy_metrics(
    metrics_history: Dict[str, List[float]],
) -> None:
    import csv

    # Write CSV file
    output_csv = "output/test_accuracy_metrics.csv"
    num_epochs = len(metrics_history["test_accuracy"])
    with open(output_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "test_accuracy", "mean_accuracy", "spread_accuracy"])
        for i in range(num_epochs):
            writer.writerow(
                [
                    i,
                    metrics_history["test_accuracy"][i],
                    metrics_history["test_accuracy_mean"][i],
                    metrics_history["test_accuracy_spread"][i],
                ]
            )
    print(f"Test accuracy metrics saved to {output_csv}")

    # Generate a professional-style plot.
    plt.style.use("seaborn-paper")
    epochs = list(range(num_epochs))
    test_acc = metrics_history["test_accuracy"]
    mean_acc = metrics_history["test_accuracy_mean"]
    spread_acc = metrics_history["test_accuracy_spread"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, test_acc, label="Test Accuracy", marker="o", linestyle="-")
    ax.plot(
        epochs,
        mean_acc,
        label="Moving Mean (last 10 epochs)",
        marker="s",
        linestyle="--",
    )
    # Fill the area between (mean - spread) and (mean + spread)
    mean_arr = np.array(mean_acc)
    spread_arr = np.array(spread_acc)
    ax.fill_between(
        epochs,
        mean_arr - spread_arr,
        mean_arr + spread_arr,
        color="gray",
        alpha=0.3,
        label="Spread (±1 std)",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Test Accuracy Metrics Over Epochs")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    output_fig = "output/test_accuracy_metrics.png"
    plt.savefig(output_fig, dpi=300)
    plt.close(fig)
    print(f"Test accuracy graph saved to {output_fig}")


def train_model(
    model: nnx.Module,
    start_epoch: int,
    metrics: nnx.MultiMetric,
    config: Dict,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    rng_key: jnp.ndarray,
) -> Dict[str, list]:
    metrics_history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "train_accuracy_mean": [],
        "train_accuracy_spread": [],
        "test_loss": [],
        "test_accuracy": [],
        "test_accuracy_mean": [],
        "test_accuracy_spread": [],
    }
    augmentation_params = AugmentationParams(**config["training"]["augmentation"])
    
    # Create optimizer once
    initial_lr = config["training"]["base_learning_rate"]
    optimizer = create_optimizer(model, initial_lr, 1e-4)

    for epoch in range(
        start_epoch, start_epoch + config["training"]["num_epochs_to_train_now"]
    ):
        learning_rate = lr_schedule(epoch, config)
        weight_decay = min(1e-4, learning_rate / 10)
        
        print(f"Epoch: {epoch}, Learning rate: {learning_rate:.6e}")

        metrics.reset()
        for batch in train_dataloader:
            batch = jax_collate(batch)
            _, dropout_rng = random.split(rng_key)
            batch = augment_data_batch(batch, dropout_rng, augmentation_params)
            train_step(model, optimizer, metrics, batch, learning_rate, weight_decay)

        # Visualize augmented images once per epoch
        visualize_augmented_images(batch, epoch, num_images=9)

        for metric, value in metrics.compute().items():
            metrics_history[f"train_{metric}"].append(value.item())
        print(
            f"[train] epoch: {epoch}, loss: {metrics_history['train_loss'][-1]:.4f}, "
            f"accuracy: {metrics_history['train_accuracy'][-1]:.4f}"
        )

        metrics.reset()
        for test_batch in test_dataloader:
            test_batch = jax_collate(test_batch)
            eval_step(model, metrics, test_batch)
        for metric, value in metrics.compute().items():
            metrics_history[f"test_{metric}"].append(value.item())
        metrics.reset()
        print(
            f"[test] epoch: {epoch}, loss: {metrics_history['test_loss'][-1]:.4f}, "
            f"accuracy: {metrics_history['test_accuracy'][-1]:.4f}"
        )

        # Compute mean and spread of the accuracy over the last 10 epochs for both train and test
        for split in ["train", "test"]:
            acc_key = f"{split}_accuracy"
            mean_key = f"{split}_accuracy_mean"
            spread_key = f"{split}_accuracy_spread"
            if len(metrics_history[acc_key]) >= 10:
                n = len(metrics_history[acc_key])
                n_recent = min(n, 10)
                recent_accuracies = metrics_history[acc_key][-n_recent:]
                mean_accuracy, spread_accuracy = compute_mean_and_spread(
                    recent_accuracies
                )
                print(
                    f"[{split}] last {n_recent} epochs mean accuracy: {mean_accuracy:.4f}, "
                    f"spread: {spread_accuracy:.4f}"
                )
                metrics_history[mean_key].append(mean_accuracy)
                metrics_history[spread_key].append(spread_accuracy)
            else:
                metrics_history[mean_key].append(0.0)
                metrics_history[spread_key].append(0.0)

        # Save test accuracy metrics at the end of every epoch
        save_test_accuracy_metrics(metrics_history, epoch)

        visualize_incorrect_classifications(model, test_dataloader, epoch)
        save_model(model, config["training"]["checkpoint_dir"], epoch)
        load_and_plot_test_accuracy_metrics(
            "output/test_accuracy_metrics.csv", "output/test_accuracy_metrics.png"
        )
    return metrics_history


def visualize_results(
    metrics_history: Dict[str, list],
    model: nnx.Module,
    test_dataloader: DataLoader,
    epoch: int,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.set_title("Loss")
    ax2.set_title("Accuracy")
    for dataset in ("train", "test"):
        ax1.plot(metrics_history[f"{dataset}_loss"], label=f"{dataset}_loss")
        ax2.plot(metrics_history[f"{dataset}_accuracy"], label=f"{dataset}_accuracy")
    ax1.legend()
    ax2.legend()
    plt.savefig(f"output/results_epoch_{epoch}.png")
    plt.close(fig)

    for test_batch in test_dataloader:
        test_batch = jax_collate(test_batch)
        break
    preds = pred_step(model, test_batch)
    fig, axs = plt.subplots(5, 5, figsize=(12, 12))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(test_batch["image"][i, ..., 0], cmap="gray")
        ax.set_title(f"Prediction: {preds[i]}, Label: {test_batch['label'][i]}")
        ax.axis("off")
    plt.savefig(f"output/prediction_example_epoch_{epoch}.png")
    plt.close(fig)


def save_model_visualization(model: nnx.Module) -> None:
    html_content = treescope.render_to_html(model)
    output_file = "treescope_output.html"
    with open(output_file, "w") as file:
        file.write(html_content)
    print(f"TreeScope HTML saved to '{output_file}'.")


def save_model(model: nnx.Module, ckpt_dir: str, epoch: int):
    state_dir = f"{ckpt_dir}/epoch_{epoch}"
    if not os.path.exists(state_dir):
        os.makedirs(state_dir)
    keys, state = nnx.state(model, nnx.RngKey, ...)
    keys = jax.tree.map(jax.random.key_data, keys)
    checkpointer = orbax.PyTreeCheckpointer()
    checkpointer.save(state_dir, state, force=True)
    zip_path = f"{ckpt_dir}/epoch_{epoch}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(state_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=state_dir)
                zipf.write(file_path, arcname)
    shutil.rmtree(state_dir)
    print(f"Model checkpoint for epoch {epoch} saved to {zip_path}")


def load_model(model: nnx.Module, ckpt_dir: str, epoch: int, seed: int) -> nnx.Module:
    zip_path = f"{ckpt_dir}/epoch_{epoch}.zip"
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Checkpoint file not found at {zip_path}")
    extract_dir = f"{ckpt_dir}/epoch_{epoch}_temp"
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zipf:
        zipf.extractall(extract_dir)
    keys, state = nnx.state(model, nnx.RngKey, ...)
    checkpointer = orbax.PyTreeCheckpointer()
    restored_state = checkpointer.restore(extract_dir, item=state)
    nnx.update(model, keys, restored_state)
    shutil.rmtree(extract_dir)
    print(f"Model checkpoint for epoch {epoch} loaded from {zip_path}")
    return model


def get_latest_checkpoint_epoch(ckpt_dir: str) -> int:
    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.exists(ckpt_dir):
        return 0
    files_and_dirs = os.listdir(ckpt_dir)
    epoch_pattern = re.compile(r"epoch_(\d+)")
    epochs = [
        int(match.group(1))
        for name in files_and_dirs
        if (match := epoch_pattern.search(name))
    ]
    return max(epochs, default=0)


def test_onnx_model(onnx_model_path: str, test_dataloader: DataLoader) -> None:
    """Test the exported ONNX model with ONNX Runtime and the MNIST test set."""
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    correct = 0
    total = 0

    for batch in test_dataloader:
        images, labels = batch
        images = images.numpy()
        labels = labels.numpy()
        images = images.transpose(0, 2, 3, 1)  # Convert to NHWC format

        inputs = {input_name: images, "deterministic": np.array(True)}
        preds = session.run([output_name], inputs)[0]
        preds = np.argmax(preds, axis=1)

        correct += (preds == labels).sum()
        total += labels.shape[0]

    accuracy = correct / total
    print(f"ONNX model test accuracy: {accuracy:.4f}")


# =============================================================================
# Main function
# =============================================================================


def main() -> None:
    os.makedirs("output", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    # load_and_plot_test_accuracy_metrics("output/test_accuracy_metrics.csv", "output/test_accuracy_metrics.png")

    # jax.config.update("jax_log_compiles", True)  # Keep commented out unless debugging

    # Define all configuration parameters in a hierarchical dictionary.
    # "seed" is now at the top level.
    config: Dict[str, Any] = {
        "seed": 5678,
        "training": {
            "enable_training": True,
            "batch_size": 64,
            "base_learning_rate": 0.0001,
            "num_epochs_to_train_now": 500,
            "warmup_epochs": 5,
            "checkpoint_dir": os.path.abspath("./data/checkpoints/"),
            "data_dir": "./data",
            "augmentation": {
                # translation in pixels
                "enable_translation": True,
                "max_translation": 3.0,
                # scaling factors in x (horizontal) and y (vertical) directions
                "enable_scaling": True,
                "scale_min_x": 0.85,
                "scale_max_x": 1.15,
                "scale_min_y": 0.85,
                "scale_max_y": 1.15,
                # rotation in degrees
                "enable_rotation": True,
                "max_rotation": 15.0,
                # elastic local deformations
                "enable_elastic": True,
                "elastic_alpha": 1.0,  # distortion intensity
                "elastic_sigma": 0.5,  # smoothing
                # rectangle erasing
                "enable_rect_erasing": False,
                "rect_erase_height": 2,
                "rect_erase_width": 20,
            },
        },
        "model": {
            "height": 28,
            "width": 28,
            "num_hiddens": 256,
            "num_layers": 4,
            "num_heads": 4,
            "mlp_dim": 256,
            "num_classes": 10,
            "embed_dims": [32, 128, 256],
            "kernel_size": 3,
            "strides": [1, 2, 2],
            "embedding_type": "conv",  # "patch" or "conv"
            "embedding_dropout_rate": 0.5,
            "attention_dropout_rate": 0.5,
            "mlp_dropout_rate": 0.5,
        },
        "onnx": {
            "model_name": "mnist_vit_model",
            "output_path": "docs/mnist_vit_model.onnx",
            "input_shapes": [("B", 28, 28, 1)],
            "input_params": {
                "deterministic": True,
            },
        },
    }

    # Set up the model's RNG using the top-level seed.
    config["model"]["rngs"] = nnx.Rngs(config["seed"])

    train_dataloader, test_dataloader = get_dataset_torch_dataloaders(
        config["training"]["batch_size"], config["training"]["data_dir"]
    )

    rngs = nnx.Rngs(config["seed"])
    rng_key = rngs.as_jax_rng()

    start_epoch = get_latest_checkpoint_epoch(config["training"]["checkpoint_dir"])
    config["training"]["start_epoch"] = start_epoch
    print(f"Resuming from epoch: {start_epoch}.")

    # Create the model using parameters from the config.
    model = VisionTransformer(**config["model"])

    if start_epoch > 0:
        try:
            model = load_model(
                model, config["training"]["checkpoint_dir"], start_epoch, config["seed"]
            )
            print(f"Loaded model from epoch {start_epoch}")
        except FileNotFoundError:
            print(
                f"Checkpoint for epoch {start_epoch} not found, starting from scratch."
            )
            start_epoch = 0
            model = VisionTransformer(**config["model"])

    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    if config["training"]["enable_training"]:
        metrics_history = train_model(
            model,
            start_epoch,
            metrics,
            config,
            train_dataloader,
            test_dataloader,
            rng_key,
        )
        visualize_results(
            metrics_history,
            model,
            test_dataloader,
            start_epoch + config["training"]["num_epochs_to_train_now"] - 1,
        )

    # onnx export
    inputs = config["onnx"]["input_shapes"]
    input_params = {"deterministic": True}
    output_path = config["onnx"]["output_path"]
    print("Exporting model to ONNX...")
    onnx_model = to_onnx(model, inputs, input_params)
    onnx.save_model(onnx_model, output_path)
    print(f"Model exported to {output_path}")

    # Test the exported ONNX model with a concrete batch size
    # Replace symbolic 'B' with a concrete batch size (e.g., 4)
    concrete_shapes = []
    for shape in inputs:
        # Convert placeholder 'B' to an actual integer batch size
        concrete_shape = tuple(4 if dim == "B" else dim for dim in shape)
        concrete_shapes.append(concrete_shape)

    xs = [jax.random.normal(rng_key, shape) for shape in concrete_shapes]

    # Correct allclose usage: pass model kwargs directly, not as jax_kwargs
    model.eval()
    result = allclose(model, output_path, xs, input_params)
    print(f"ONNX allclose result: {result}")

    # Also test with actual test data from the dataloader
    test_onnx_model(output_path, test_dataloader)


if __name__ == "__main__":
    main()
