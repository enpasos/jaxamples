# file: jaxamples/mnist_vit.py

import os
import re
import shutil
import zipfile
import warnings
from typing import Dict, Tuple, List, Any

import jax
import jax.numpy as jnp
from jax import Array
import optax
import torchvision
import treescope
from flax import nnx
from jax import random
from jax.image import scale_and_translate
from jax.scipy.ndimage import map_coordinates
from torch.utils.data import DataLoader
from torchvision import transforms
import jax2onnx.plugins  # noqa: F401
from jax2onnx.examples.mnist_vit import VisionTransformer
import orbax.checkpoint as orbax
from jax2onnx.to_onnx import to_onnx
import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use a non-interactive backend to avoid Tkinter-related issues
import matplotlib.pyplot as plt

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


@jax.jit
def augment_data_batch(
    batch: Dict[str, jnp.ndarray], rng_key: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """Augments a batch of images."""
    images = batch["image"]
    batch_size, height, width, channels = images.shape

    def augment_single_image(image, key):
        key1, key2, key3, key4, key5 = random.split(key, 5)
        max_translation = 2.0
        tx = random.uniform(key1, minval=-max_translation, maxval=max_translation)
        ty = random.uniform(key2, minval=-max_translation, maxval=max_translation)
        translation = jnp.array([ty, tx])
        scale_factor_x = random.uniform(key3, minval=0.7, maxval=1.1)
        scale_factor_y = random.uniform(key4, minval=0.8, maxval=1.1)
        scale = jnp.array([scale_factor_y, scale_factor_x])
        max_rotation = 10.0 * (jnp.pi / 180.0)
        rotation_angle = random.uniform(key5, minval=-max_rotation, maxval=max_rotation)
        rotated_image = rotate_image(image, jnp.rad2deg(rotation_angle))
        augmented_image = scale_and_translate(
            image=rotated_image,
            shape=(height, width, channels),
            spatial_dims=(0, 1),
            scale=scale,
            translation=translation,
            method="linear",
            antialias=True,
        )
        return augmented_image

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
    optimizer.update(grads, learning_rate=learning_rate, weight_decay=weight_decay)


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
    fig, axes = (
        plt.subplots(1, num_images, figsize=(15, 5))
        if num_images > 1
        else (plt.subplots(1, num_images, figsize=figsize)[1],)
    )
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
    return nnx.Optimizer(
        model, optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    )


# =============================================================================
# Training, evaluation, checkpointing and visualization functions
# =============================================================================


def compute_mean_and_spread(values: List[float]) -> Tuple[float, float]:
    """Compute the mean and spread (standard deviation) of a list of values."""
    mean = np.mean(values)
    spread = np.std(values)
    return mean, spread


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
        "test_loss": [],
        "test_accuracy": [],
    }
    optimizer = create_optimizer(model, config["training"]["base_learning_rate"], 1e-4)

    for epoch in range(
        start_epoch, start_epoch + config["training"]["num_epochs_to_train_now"]
    ):
        learning_rate = lr_schedule(epoch, config)
        print(f"Epoch: {epoch}, Learning rate: {learning_rate}")
        weight_decay = min(1e-4, learning_rate / 10)

        metrics.reset()
        for batch in train_dataloader:
            batch = jax_collate(batch)
            _, dropout_rng = random.split(rng_key)
            batch = augment_data_batch(batch, dropout_rng)
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

        # Compute mean and spread of the accuracy over the last 10 epochs
        if len(metrics_history["test_accuracy"]) >= 10:
            recent_accuracies = metrics_history["test_accuracy"][-10:]
            mean_accuracy, spread_accuracy = compute_mean_and_spread(recent_accuracies)
            print(
                f"[test] last 10 epochs mean accuracy: {mean_accuracy:.4f}, "
                f"spread: {spread_accuracy:.4f}"
            )

        visualize_incorrect_classifications(model, test_dataloader, epoch)
        save_model(model, config["training"]["checkpoint_dir"], epoch)
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


# =============================================================================
# Main function
# =============================================================================


def main() -> None:
    os.makedirs("output", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    # jax.config.update("jax_log_compiles", True)  # Keep commented out unless debugging

    # Define all configuration parameters in a hierarchical dictionary.
    # "seed" is now at the top level.
    config: Dict[str, Any] = {
        "seed": 5678,
        "training": {
            "batch_size": 64,
            "base_learning_rate": 0.0001,
            "num_epochs_to_train_now": 200,
            "warmup_epochs": 5,
            "checkpoint_dir": os.path.abspath("./data/checkpoints/"),
            "data_dir": "./data",
        },
        "model": {
            "height": 28,
            "width": 28,
            "num_hiddens": 256,
            "num_layers": 8,
            "num_heads": 4,
            "mlp_dim": 256,
            "num_classes": 10,
            "dropout_rate": 0.5,
            "embed_dims": [32, 128, 256],
            "kernel_size": 3,
            "strides": [1, 2, 2],
        },
        "onnx": {
            "model_file_name": "mnist_vit_model.onnx",
            "output_path": "docs/mnist_vit_model.onnx",
            "input_shapes": [(1, 28, 28, 1)],
            "params": {
                "pre_transpose": [(0, 3, 1, 2)],  # Convert JAX → ONNX
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

    metrics_history = train_model(
        model, start_epoch, metrics, config, train_dataloader, test_dataloader, rng_key
    )
    visualize_results(
        metrics_history,
        model,
        test_dataloader,
        start_epoch + config["training"]["num_epochs_to_train_now"] - 1,
    )

    config["onnx"]["component"] = model
    print("Exporting model to ONNX...")
    to_onnx(**config["onnx"])


if __name__ == "__main__":
    main()
