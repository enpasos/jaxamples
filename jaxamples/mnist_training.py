import csv
import functools
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.lax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
import torchvision
import treescope
from flax import nnx, serialization
from flax.struct import dataclass, field
from jax import Array, random
from jax.image import scale_and_translate
from jax.scipy.ndimage import map_coordinates
from torch.utils.data import DataLoader

from jaxamples.mnist_config import MnistExampleConfig, TrainingConfig
from jaxamples import mnist_onnx_eval
from jaxamples.mnist_data import get_mnist_transform


test_onnx_model = mnist_onnx_eval.test_onnx_model

TrainingConfigInput = MnistExampleConfig | TrainingConfig | Dict[str, Any]


def _get_training_section(config: TrainingConfigInput) -> TrainingConfig | Dict[str, Any]:
    if isinstance(config, MnistExampleConfig):
        return config.training
    if isinstance(config, TrainingConfig):
        return config
    return config["training"]


def _get_training_value(config: TrainingConfigInput, key: str) -> Any:
    training = _get_training_section(config)
    if isinstance(training, TrainingConfig):
        return getattr(training, key)
    return training[key]


def _get_augmentation_dict(config: TrainingConfigInput) -> Dict[str, Any]:
    augmentation = _get_training_value(config, "augmentation")
    if hasattr(augmentation, "to_dict"):
        return augmentation.to_dict()
    return augmentation


def _get_output_dir(config: TrainingConfigInput) -> str:
    if isinstance(config, MnistExampleConfig):
        return str(config.artifact_dir())

    training = _get_training_section(config)
    if isinstance(training, TrainingConfig) and training.output_dir:
        return training.output_dir
    if isinstance(training, dict) and training.get("output_dir"):
        return training["output_dir"]
    return "output"


def _get_metrics_paths(config: TrainingConfigInput) -> Tuple[str, str]:
    if isinstance(config, MnistExampleConfig):
        return str(config.metrics_csv_path()), str(config.metrics_plot_path())

    output_dir = _get_output_dir(config)
    return (
        str(Path(output_dir) / "test_accuracy_metrics.csv"),
        str(Path(output_dir) / "test_accuracy_metrics.png"),
    )


def get_dataset_torch_dataloaders(batch_size: int, data_dir: str = "./data"):
    transform = get_mnist_transform()
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


def _get_pyplot():
    if "MPLBACKEND" not in os.environ and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _ensure_parent_dir(path: str | Path) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_path


def _build_output_path(filename: str, output_dir: str = "output") -> Path:
    return _ensure_parent_dir(Path(output_dir) / filename)


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
    ds: Dict[str, jnp.ndarray],
    epoch: int,
    num_images: int = 9,
    output_dir: str = "output",
) -> None:
    num_available_images = int(ds["image"].shape[0])
    num_images = min(num_images, num_available_images)
    if num_images == 0:
        print("No augmented images to visualize.")
        return

    plt = _get_pyplot()
    fig, axes = plt.subplots(1, num_images, figsize=(max(1, num_images) * 1.8, 5))
    if num_images == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(ds["image"][i, ..., 0], cmap="gray")
        ax.axis("off")

    output_path = _build_output_path(f"augmented_images_epoch{epoch}.png", output_dir)
    plt.savefig(output_path)
    plt.close(fig)


@dataclass
class AugmentationParams:
    max_translation: float
    scale_min_x: float
    scale_max_x: float
    scale_min_y: float
    scale_max_y: float
    max_rotation: float
    elastic_alpha: float
    elastic_sigma: float

    enable_elastic: bool = field(pytree_node=False)
    enable_rotation: bool = field(pytree_node=False)
    enable_scaling: bool = field(pytree_node=False)
    enable_translation: bool = field(pytree_node=False)
    enable_rect_erasing: bool = field(pytree_node=False)
    rect_erase_height: int = field(pytree_node=False)
    rect_erase_width: int = field(pytree_node=False)


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

        if getattr(augmentation_params, "enable_rect_erasing", False):
            erase_h = getattr(augmentation_params, "rect_erase_height", 6)
            erase_w = getattr(augmentation_params, "rect_erase_width", 6)
            max_y = height - erase_h
            max_x = width - erase_w
            y0 = jax.lax.floor(
                random.uniform(key7, (), minval=0, maxval=max_y + 1)
            ).astype(jnp.int32)
            x0 = jax.lax.floor(
                random.uniform(key8, (), minval=0, maxval=max_x + 1)
            ).astype(jnp.int32)
            mask = jnp.ones_like(image)
            erase_shape = (erase_h, erase_w, channels)
            zero_patch = jnp.zeros(erase_shape, dtype=image.dtype)
            mask = jax.lax.dynamic_update_slice(mask, zero_patch, (y0, x0, 0))
            image = image * mask
        return image

    rng_keys = random.split(rng_key, num=batch_size)
    augmented_images = jax.vmap(augment_single_image)(images, rng_keys)
    return {"image": augmented_images, "label": batch["label"]}


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
    optimizer.opt_state.hyperparams["learning_rate"] = learning_rate
    optimizer.opt_state.hyperparams["weight_decay"] = weight_decay
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
    max_images: int = 50,
    output_dir: str = "output",
) -> None:
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

    if incorrect_images:
        incorrect_images = jnp.concatenate(incorrect_images, axis=0)
        incorrect_images_typed: Array = incorrect_images
        incorrect_labels = jnp.concatenate(incorrect_labels, axis=0)
        incorrect_preds = jnp.concatenate(incorrect_preds, axis=0)
    else:
        print("No incorrect classifications found.")
        return

    num_images = len(incorrect_images)
    if num_images > max_images:
        print(
            f"Too many incorrect classifications ({num_images}). "
            f"Showing the first {max_images}."
        )
        incorrect_images_typed = incorrect_images_typed[:max_images]
        incorrect_labels = incorrect_labels[:max_images]
        incorrect_preds = incorrect_preds[:max_images]
        num_images = max_images

    plt = _get_pyplot()
    num_cols = min(10, num_images)
    num_rows = math.ceil(num_images / num_cols)
    if figsize == (15, 5):
        figsize = (max(6, num_cols * 1.8), max(3, num_rows * 2.2))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = np.atleast_1d(axes).reshape(num_rows, num_cols)
    for i, ax in enumerate(axes.flatten()):
        if i >= num_images:
            ax.axis("off")
            continue
        ax.imshow(incorrect_images_typed[i, ..., 0], cmap="gray")
        ax.set_title(f"{incorrect_labels[i]}\nbut\n{incorrect_preds[i]}")
        ax.axis("off")

    output_path = _build_output_path(
        f"incorrect_classifications_epoch{epoch}.png", output_dir
    )
    plt.savefig(output_path)
    plt.close(fig)


def lr_schedule(epoch: int, config: TrainingConfigInput) -> float:
    total_epochs = _get_training_value(config, "start_epoch") + _get_training_value(
        config, "num_epochs_to_train_now"
    )
    base_lr = _get_training_value(config, "base_learning_rate")
    return 0.5 * base_lr * (1 + jnp.cos(jnp.pi * epoch / total_epochs))


def create_optimizer(
    model: nnx.Module, learning_rate: float, weight_decay: float
) -> nnx.Optimizer:
    tx = optax.inject_hyperparams(optax.adamw)(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def compute_mean_and_spread(values: List[float]) -> Tuple[float, float]:
    mean = np.mean(values)
    spread = np.std(values)
    return mean, spread


def save_test_accuracy_metrics(
    metrics_history: Dict[str, List[float]],
    epoch: int,
    output_csv: str = "output/test_accuracy_metrics.csv",
) -> None:
    output_csv_path = _ensure_parent_dir(output_csv)
    file_exists = output_csv_path.is_file()
    with output_csv_path.open(mode="a", newline="") as csv_file:
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
    print(
        f"Test and train accuracy metrics for epoch {epoch} saved to {output_csv_path}"
    )


def load_and_plot_test_accuracy_metrics(csv_filepath: str, output_fig: str) -> None:
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
    plt = _get_pyplot()
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
    output_fig_path = _ensure_parent_dir(output_fig)
    plt.savefig(output_fig_path, dpi=300)
    plt.close(fig)
    print(f"Test and train accuracy graph saved to {output_fig_path}")


def save_and_plot_test_accuracy_metrics(
    metrics_history: Dict[str, List[float]],
    output_csv: str = "output/test_accuracy_metrics.csv",
    output_fig: str = "output/test_accuracy_metrics.png",
) -> None:
    output_csv_path = _ensure_parent_dir(output_csv)
    num_epochs = len(metrics_history["test_accuracy"])
    with output_csv_path.open(mode="w", newline="") as csv_file:
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
    print(f"Test accuracy metrics saved to {output_csv_path}")

    plt = _get_pyplot()
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
    output_fig_path = _ensure_parent_dir(output_fig)
    plt.savefig(output_fig_path, dpi=300)
    plt.close(fig)
    print(f"Test accuracy graph saved to {output_fig_path}")


def train_model(
    model: nnx.Module,
    start_epoch: int,
    metrics: nnx.MultiMetric,
    config: TrainingConfigInput,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    rng_key: jnp.ndarray,
    *,
    augmentation_params_cls: type[AugmentationParams] = AugmentationParams,
    create_optimizer_fn: Callable[[nnx.Module, float, float], nnx.Optimizer] = create_optimizer,
    lr_schedule_fn: Callable[[int, TrainingConfigInput], float] = lr_schedule,
    jax_collate_fn: Callable[[Any], Dict[str, jnp.ndarray]] = jax_collate,
    augment_data_batch_fn: Callable[
        [Dict[str, jnp.ndarray], jnp.ndarray, AugmentationParams],
        Dict[str, jnp.ndarray],
    ] = augment_data_batch,
    train_step_fn: Callable[..., None] = train_step,
    eval_step_fn: Callable[..., None] = eval_step,
    visualize_augmented_images_fn: Callable[..., None] = visualize_augmented_images,
    compute_mean_and_spread_fn: Callable[[List[float]], Tuple[float, float]] = compute_mean_and_spread,
    save_test_accuracy_metrics_fn: Callable[..., None] = save_test_accuracy_metrics,
    visualize_incorrect_classifications_fn: Callable[..., None] = visualize_incorrect_classifications,
    save_model_fn: Callable[[nnx.Module, str, int], None] = None,
    load_and_plot_test_accuracy_metrics_fn: Callable[[str, str], None] = load_and_plot_test_accuracy_metrics,
) -> Dict[str, list]:
    if save_model_fn is None:
        save_model_fn = save_model

    output_dir = _get_output_dir(config)
    metrics_csv_path, metrics_fig_path = _get_metrics_paths(config)

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
    augmentation_params = augmentation_params_cls(**_get_augmentation_dict(config))

    initial_lr = _get_training_value(config, "base_learning_rate")
    optimizer = create_optimizer_fn(model, initial_lr, 1e-4)

    for epoch in range(
        start_epoch, start_epoch + _get_training_value(config, "num_epochs_to_train_now")
    ):
        learning_rate = lr_schedule_fn(epoch, config)
        weight_decay = min(1e-4, learning_rate / 10)

        print(f"Epoch: {epoch}, Learning rate: {learning_rate:.6e}")

        metrics.reset()
        last_train_batch = None
        for batch in train_dataloader:
            batch = jax_collate_fn(batch)
            rng_key, dropout_rng = random.split(rng_key)
            batch = augment_data_batch_fn(batch, dropout_rng, augmentation_params)
            train_step_fn(model, optimizer, metrics, batch, learning_rate, weight_decay)
            last_train_batch = batch

        if last_train_batch is None:
            raise ValueError(
                "train_dataloader did not yield any batches. "
                "Reduce batch_size or disable drop_last."
            )

        visualize_augmented_images_fn(
            last_train_batch, epoch, num_images=9, output_dir=output_dir
        )

        for metric, value in metrics.compute().items():
            metrics_history[f"train_{metric}"].append(value.item())
        print(
            f"[train] epoch: {epoch}, loss: {metrics_history['train_loss'][-1]:.4f}, "
            f"accuracy: {metrics_history['train_accuracy'][-1]:.4f}"
        )

        metrics.reset()
        num_test_batches = 0
        for test_batch in test_dataloader:
            test_batch = jax_collate_fn(test_batch)
            eval_step_fn(model, metrics, test_batch)
            num_test_batches += 1
        if num_test_batches == 0:
            raise ValueError(
                "test_dataloader did not yield any batches. "
                "Reduce batch_size or disable drop_last."
            )
        for metric, value in metrics.compute().items():
            metrics_history[f"test_{metric}"].append(value.item())
        metrics.reset()
        print(
            f"[test] epoch: {epoch}, loss: {metrics_history['test_loss'][-1]:.4f}, "
            f"accuracy: {metrics_history['test_accuracy'][-1]:.4f}"
        )

        for split in ["train", "test"]:
            acc_key = f"{split}_accuracy"
            mean_key = f"{split}_accuracy_mean"
            spread_key = f"{split}_accuracy_spread"
            if len(metrics_history[acc_key]) >= 10:
                n_recent = min(len(metrics_history[acc_key]), 10)
                recent_accuracies = metrics_history[acc_key][-n_recent:]
                mean_accuracy, spread_accuracy = compute_mean_and_spread_fn(
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

        save_test_accuracy_metrics_fn(
            metrics_history, epoch, output_csv=metrics_csv_path
        )
        visualize_incorrect_classifications_fn(
            model, test_dataloader, epoch, output_dir=output_dir
        )
        save_model_fn(model, _get_training_value(config, "checkpoint_dir"), epoch)
        load_and_plot_test_accuracy_metrics_fn(metrics_csv_path, metrics_fig_path)
    return metrics_history


def visualize_results(
    metrics_history: Dict[str, list],
    model: nnx.Module,
    test_dataloader: DataLoader,
    epoch: int,
    output_dir: str = "output",
    *,
    jax_collate_fn: Callable[[Any], Dict[str, jnp.ndarray]] = jax_collate,
    pred_step_fn: Callable[[nnx.Module, Dict[str, jnp.ndarray]], jnp.ndarray] = pred_step,
):
    plt = _get_pyplot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.set_title("Loss")
    ax2.set_title("Accuracy")
    for dataset in ("train", "test"):
        ax1.plot(metrics_history[f"{dataset}_loss"], label=f"{dataset}_loss")
        ax2.plot(metrics_history[f"{dataset}_accuracy"], label=f"{dataset}_accuracy")
    ax1.legend()
    ax2.legend()
    metrics_output_path = _build_output_path(f"results_epoch_{epoch}.png", output_dir)
    plt.savefig(metrics_output_path)
    plt.close(fig)

    first_test_batch = None
    for test_batch in test_dataloader:
        first_test_batch = jax_collate_fn(test_batch)
        break
    if first_test_batch is None:
        print("No test batches available for prediction visualization. Skipping.")
        return

    preds = pred_step_fn(model, first_test_batch)
    num_examples = min(25, int(first_test_batch["image"].shape[0]))
    fig, axs = plt.subplots(5, 5, figsize=(12, 12))
    for i, ax in enumerate(axs.flatten()):
        if i >= num_examples:
            ax.axis("off")
            continue
        ax.imshow(first_test_batch["image"][i, ..., 0], cmap="gray")
        ax.set_title(f"Prediction: {preds[i]}, Label: {first_test_batch['label'][i]}")
        ax.axis("off")
    prediction_output_path = _build_output_path(
        f"prediction_example_epoch_{epoch}.png", output_dir
    )
    plt.savefig(prediction_output_path)
    plt.close(fig)


def save_model_visualization(
    model: nnx.Module, output_file: str = "treescope_output.html"
) -> None:
    html_content = treescope.render_to_html(model)
    output_path = _ensure_parent_dir(output_file)
    with output_path.open("w") as file:
        file.write(html_content)
    print(f"TreeScope HTML saved to '{output_path}'.")


CKPT_EXTENSION = ".msgpack"


def _to_numpy_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    def _convert(x):
        value = jax.device_get(x)
        try:
            value = jax.random.key_data(value)
        except TypeError:
            pass
        return np.asarray(value)

    return jax.tree_util.tree_map(_convert, tree)


def save_model(model: nnx.Module, ckpt_dir: str, epoch: int):
    os.makedirs(ckpt_dir, exist_ok=True)
    keys_state, params_state = nnx.state(model, nnx.RngKey, ...)
    payload = {
        "keys": _to_numpy_tree(nnx.to_pure_dict(keys_state)),
        "state": _to_numpy_tree(nnx.to_pure_dict(params_state)),
    }
    ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}{CKPT_EXTENSION}")
    with open(ckpt_path, "wb") as fout:
        fout.write(serialization.to_bytes(payload))
    print(f"Model checkpoint for epoch {epoch} saved to {ckpt_path}")


def load_model(model: nnx.Module, ckpt_dir: str, epoch: int, seed: int) -> nnx.Module:
    del seed
    ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}{CKPT_EXTENSION}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")

    with open(ckpt_path, "rb") as fin:
        raw_bytes = fin.read()

    keys_state, params_state = nnx.state(model, nnx.RngKey, ...)
    template = {
        "keys": nnx.to_pure_dict(keys_state),
        "state": nnx.to_pure_dict(params_state),
    }
    restored_payload = serialization.from_bytes(template, raw_bytes)
    restored_keys = jax.tree_util.tree_map(
        lambda x: jax.random.wrap_key_data(x)
        if isinstance(x, np.ndarray)
        and x.dtype == np.uint32
        and x.shape
        and x.shape[-1] == 2
        else x,
        restored_payload["keys"],
    )
    nnx.replace_by_pure_dict(keys_state, restored_keys)
    nnx.replace_by_pure_dict(params_state, restored_payload["state"])
    nnx.update(model, keys_state, params_state)
    print(f"Model checkpoint for epoch {epoch} loaded from {ckpt_path}")
    return model


def get_latest_checkpoint_epoch(ckpt_dir: str) -> Optional[int]:
    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.exists(ckpt_dir):
        return None
    files_and_dirs = os.listdir(ckpt_dir)
    epoch_pattern = re.compile(r"epoch_(\d+)")
    epochs = [
        int(match.group(1))
        for name in files_and_dirs
        if (match := epoch_pattern.search(name))
        and name.endswith(CKPT_EXTENSION)
    ]
    return max(epochs) if epochs else None


def resolve_checkpoint_resume(ckpt_dir: str) -> Tuple[Optional[int], int]:
    latest_checkpoint_epoch = get_latest_checkpoint_epoch(ckpt_dir)
    start_epoch = 0 if latest_checkpoint_epoch is None else latest_checkpoint_epoch + 1
    return latest_checkpoint_epoch, start_epoch
