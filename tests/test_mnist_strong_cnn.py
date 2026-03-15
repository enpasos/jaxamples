from pathlib import Path

import jax.numpy as jnp
import pytest
import torch
from flax import nnx
from torch.utils.data import DataLoader, TensorDataset

from jaxamples import mnist_cnn
from jaxamples import mnist_strong_cnn
from jaxamples import mnist_strong_cnn_run_onnx
from jaxamples import mnist_training


def get_dummy_dataloaders(batch_size: int):
    images = torch.ones(8, 1, 28, 28, dtype=torch.float32)
    labels = torch.arange(8, dtype=torch.int64) % 10
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, dataloader


def test_mnist_strong_cnn_classifier_forward_shape():
    model = mnist_strong_cnn.MnistStrongCnnClassifier(
        height=28,
        width=28,
        num_classes=10,
        conv_channels=[16, 32, 64, 96],
        dense_hidden_dim=64,
        rngs=nnx.Rngs(0),
    )

    logits = model(jnp.ones((4, 28, 28, 1), dtype=jnp.float32), deterministic=True)

    assert logits.shape == (4, 10)
    assert logits.dtype == jnp.float32


def test_mnist_strong_cnn_train_step_runs_under_jit():
    model = mnist_strong_cnn.MnistStrongCnnClassifier(
        height=28,
        width=28,
        num_classes=10,
        conv_channels=[16, 32, 64, 96],
        dense_hidden_dim=64,
        rngs=nnx.Rngs(0),
    )
    optimizer = mnist_training.create_optimizer(
        model, learning_rate=0.001, weight_decay=1e-4
    )
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )
    batch = {
        "image": jnp.ones((2, 28, 28, 1), dtype=jnp.float32),
        "label": jnp.array([0, 1], dtype=jnp.int32),
    }

    mnist_training.train_step(
        model,
        optimizer,
        metrics,
        batch,
        learning_rate=0.001,
        weight_decay=1e-4,
    )

    computed = metrics.compute()
    assert computed["loss"].shape == ()
    assert computed["accuracy"].shape == ()


def test_mnist_strong_cnn_default_config_uses_shared_training_setup():
    config = mnist_strong_cnn.get_default_config()
    cnn_config = mnist_cnn.get_default_config()

    assert config.training.base_learning_rate == pytest.approx(1e-4)
    assert config.training.weight_decay == pytest.approx(1e-4)
    assert config.training.checkpoint_dir.endswith("strong_cnn_residual_ln_checkpoints")
    assert config.training.augmentation.to_dict() == cnn_config.training.augmentation.to_dict()
    assert config.model.conv_channels == [32, 64, 128, 192]
    assert config.model.dense_hidden_dim == 256
    assert config.model.feature_dropout_rate == pytest.approx(0.05)
    assert config.model.classifier_dropout_rate == pytest.approx(0.15)


def test_mnist_strong_cnn_main_reuses_shared_pipeline(monkeypatch, tmp_path):
    calls = {}
    train_dataloader, test_dataloader = get_dummy_dataloaders(batch_size=8)

    config = mnist_strong_cnn.get_default_config()
    config.training.num_epochs_to_train_now = 1
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")
    config.training.data_dir = str(tmp_path / "data")
    config.training.output_dir = str(tmp_path / "output")
    config.onnx.output_path = str(tmp_path / "docs" / "mnist_strong_cnn_model.onnx")

    monkeypatch.setattr(mnist_strong_cnn, "get_default_config", lambda: config)

    def fake_get_dataset_torch_dataloaders(batch_size, data_dir):
        calls["loader_args"] = {"batch_size": batch_size, "data_dir": data_dir}
        return train_dataloader, test_dataloader

    monkeypatch.setattr(
        mnist_strong_cnn,
        "get_dataset_torch_dataloaders",
        fake_get_dataset_torch_dataloaders,
    )
    monkeypatch.setattr(mnist_strong_cnn, "resolve_checkpoint_resume", lambda ckpt_dir: (None, 0))

    def fake_train_model(
        model, start_epoch, metrics, config, train_loader, test_loader, rng_key
    ):
        calls["train_model"] = {
            "start_epoch": start_epoch,
            "train_batches": len(train_loader),
            "test_batches": len(test_loader),
        }
        return {
            "train_loss": [1.0],
            "test_loss": [1.0],
            "train_accuracy": [0.8],
            "test_accuracy": [0.9],
        }

    def fake_visualize_results(metrics_history, model, test_loader, epoch, output_dir="output"):
        calls["visualize_results"] = {
            "epoch": epoch,
            "test_batches": len(test_loader),
            "output_dir": output_dir,
        }

    def fake_to_onnx(model, input_shapes, input_params):
        calls["to_onnx"] = {
            "input_shapes": input_shapes,
            "input_params": input_params,
        }
        return object()

    def fake_save_model(model_proto, output_path):
        calls["save_model"] = str(output_path)

    def fake_test_onnx_model(output_path, test_loader):
        calls["test_onnx_model"] = {
            "output_path": output_path,
            "test_batches": len(test_loader),
        }

    monkeypatch.setattr(mnist_strong_cnn, "train_model", fake_train_model)
    monkeypatch.setattr(mnist_strong_cnn, "visualize_results", fake_visualize_results)
    monkeypatch.setattr(mnist_strong_cnn, "to_onnx", fake_to_onnx)
    monkeypatch.setattr(mnist_strong_cnn.onnx, "save_model", fake_save_model)
    monkeypatch.setattr(mnist_strong_cnn, "allclose", lambda *args, **kwargs: True)
    monkeypatch.setattr(mnist_strong_cnn, "test_onnx_model", fake_test_onnx_model)

    mnist_strong_cnn.main(
        [
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--onnx-output",
            str(tmp_path / "docs" / "mnist_strong_cnn_model.onnx"),
        ]
    )

    assert calls["train_model"]["start_epoch"] == 0
    assert calls["loader_args"] == {
        "batch_size": 8,
        "data_dir": str(tmp_path / "data"),
    }
    assert calls["to_onnx"]["input_shapes"] == [("B", 28, 28, 1)]
    assert calls["to_onnx"]["input_params"] == {"deterministic": True}
    assert calls["save_model"] == str(tmp_path / "docs" / "mnist_strong_cnn_model.onnx")
    assert calls["visualize_results"]["output_dir"] == str(tmp_path / "output")
    assert calls["test_onnx_model"]["output_path"] == str(
        tmp_path / "docs" / "mnist_strong_cnn_model.onnx"
    )
    assert Path(tmp_path / "docs" / "mnist_strong_cnn_model_config.json").exists()
    benchmark_memory_path = tmp_path / "output" / "benchmark_memory.jsonl"
    assert benchmark_memory_path.exists()
    assert '"model_name":"mnist_strong_cnn_model"' in benchmark_memory_path.read_text(
        encoding="utf-8"
    )


def test_mnist_strong_cnn_run_onnx_uses_default_path(monkeypatch):
    captured = {}

    def fake_main(*, args=None, description, default_onnx_model):
        captured["args"] = args
        captured["description"] = description
        captured["default_onnx_model"] = default_onnx_model

    monkeypatch.setattr(mnist_strong_cnn_run_onnx, "run_mnist_onnx_main", fake_main)

    mnist_strong_cnn_run_onnx.main()

    assert captured["args"] is None
    assert captured["description"] == "Test ONNX MNIST strong CNN model"
    assert captured["default_onnx_model"] == "onnx/mnist_strong_cnn_model.onnx"
