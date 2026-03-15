from pathlib import Path

import jax.numpy as jnp
import pytest
import torch
from flax import nnx
from torch.utils.data import DataLoader, TensorDataset

from jaxamples import mnist_dinov3
from jaxamples import mnist_dinov3_run_onnx
from jaxamples import mnist_training


def get_dummy_dataloaders(batch_size: int):
    images = torch.ones(8, 1, 28, 28, dtype=torch.float32)
    labels = torch.arange(8, dtype=torch.int64) % 10
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, dataloader


def test_prepare_dinov3_inputs_repeats_grayscale_channels():
    images = jnp.arange(2 * 28 * 28, dtype=jnp.float32).reshape(2, 28, 28, 1)

    converted = mnist_dinov3.prepare_dinov3_inputs(images, expected_size=28)

    assert converted.shape == (2, 3, 28, 28)
    assert jnp.array_equal(converted[:, 0], converted[:, 1])
    assert jnp.array_equal(converted[:, 1], converted[:, 2])
    assert jnp.array_equal(converted[:, 0], images[..., 0])


def test_prepare_dinov3_inputs_rejects_invalid_channel_count():
    images = jnp.ones((2, 28, 28, 2), dtype=jnp.float32)

    with pytest.raises(ValueError, match="Expected 1 or 3 channels"):
        mnist_dinov3.prepare_dinov3_inputs(images, expected_size=28)


def test_mnist_dinov3_classifier_forward_shape():
    model = mnist_dinov3.MnistDinoV3Classifier(
        img_size=28,
        patch_size=7,
        embed_dim=96,
        depth=2,
        num_heads=3,
        num_classes=10,
        rngs=nnx.Rngs(0),
    )

    logits = model(jnp.ones((4, 28, 28, 1), dtype=jnp.float32), deterministic=True)

    assert logits.shape == (4, 10)
    assert logits.dtype == jnp.float32


def test_mnist_dinov3_train_step_runs_under_jit():
    model = mnist_dinov3.MnistDinoV3Classifier(
        img_size=28,
        patch_size=7,
        embed_dim=96,
        depth=2,
        num_heads=3,
        num_classes=10,
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


def test_mnist_dinov3_config_rejects_non_divisible_patch_size():
    config = mnist_dinov3.get_default_config()
    config.model.patch_size = 6

    with pytest.raises(ValueError, match="img_size must be divisible by patch_size"):
        config.validate()


def test_mnist_dinov3_default_config_uses_fairer_budget():
    config = mnist_dinov3.get_default_config()

    assert config.training.num_epochs_to_train_now == 500
    assert config.training.weight_decay == pytest.approx(1e-4)
    assert config.training.checkpoint_dir.endswith(
        "dinov3_p4_dim192_d4_h6_cls_mean_checkpoints"
    )
    assert config.model.patch_size == 4
    assert config.model.embed_dim == 192
    assert config.model.depth == 4
    assert config.model.num_heads == 6
    assert config.model.head_hidden_dim == 192
    assert config.model.head_dropout_rate == pytest.approx(0.1)
    assert config.model.pool_features == "cls_mean"


def test_lr_schedule_applies_warmup_before_cosine_decay():
    config = mnist_dinov3.get_default_config()
    config.training.start_epoch = 0
    config.training.num_epochs_to_train_now = 20
    config.training.base_learning_rate = 1e-4
    config.training.warmup_epochs = 5

    lrs = [float(mnist_training.lr_schedule(epoch, config)) for epoch in range(7)]

    assert lrs[0] == pytest.approx(2e-5)
    assert lrs[4] == pytest.approx(1e-4)
    assert lrs[5] == pytest.approx(1e-4)
    assert lrs[6] < lrs[5]


def test_lr_schedule_does_not_restart_warmup_on_resume():
    config = mnist_dinov3.get_default_config()
    config.training.start_epoch = 20
    config.training.num_epochs_to_train_now = 10
    config.training.base_learning_rate = 1e-4
    config.training.warmup_epochs = 5

    resumed_lr = float(mnist_training.lr_schedule(20, config))

    assert resumed_lr < 1e-4


def test_load_model_reports_incompatible_checkpoint(tmp_path):
    old_model = mnist_dinov3.MnistDinoV3Classifier(
        img_size=28,
        patch_size=7,
        embed_dim=96,
        depth=2,
        num_heads=3,
        num_classes=10,
        rngs=nnx.Rngs(0),
    )
    mnist_training.save_model(old_model, str(tmp_path), epoch=0)

    new_model = mnist_dinov3.create_model(mnist_dinov3.get_default_config().model, seed=0)

    with pytest.raises(
        mnist_training.IncompatibleCheckpointError, match="incompatible with the current model"
    ):
        mnist_training.load_model(new_model, str(tmp_path), epoch=0, seed=0)


def test_mnist_dinov3_main_reuses_shared_pipeline(monkeypatch, tmp_path):
    calls = {}
    train_dataloader, test_dataloader = get_dummy_dataloaders(batch_size=8)

    config = mnist_dinov3.get_default_config()
    config.training.num_epochs_to_train_now = 1
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")
    config.training.data_dir = str(tmp_path / "data")
    config.training.output_dir = str(tmp_path / "output")
    config.onnx.output_path = str(tmp_path / "docs" / "mnist_dinov3_model.onnx")

    monkeypatch.setattr(mnist_dinov3, "get_default_config", lambda: config)
    def fake_get_dataset_torch_dataloaders(batch_size, data_dir):
        calls["loader_args"] = {"batch_size": batch_size, "data_dir": data_dir}
        return train_dataloader, test_dataloader

    monkeypatch.setattr(
        mnist_dinov3,
        "get_dataset_torch_dataloaders",
        fake_get_dataset_torch_dataloaders,
    )
    monkeypatch.setattr(mnist_dinov3, "resolve_checkpoint_resume", lambda ckpt_dir: (None, 0))

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
            "train_accuracy": [0.5],
            "test_accuracy": [0.5],
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

    monkeypatch.setattr(mnist_dinov3, "train_model", fake_train_model)
    monkeypatch.setattr(mnist_dinov3, "visualize_results", fake_visualize_results)
    monkeypatch.setattr(mnist_dinov3, "to_onnx", fake_to_onnx)
    monkeypatch.setattr(mnist_dinov3.onnx, "save_model", fake_save_model)
    monkeypatch.setattr(mnist_dinov3, "allclose", lambda *args, **kwargs: True)
    monkeypatch.setattr(mnist_dinov3, "test_onnx_model", fake_test_onnx_model)

    mnist_dinov3.main(
        [
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--onnx-output",
            str(tmp_path / "docs" / "mnist_dinov3_model.onnx"),
        ]
    )

    assert calls["train_model"]["start_epoch"] == 0
    assert calls["loader_args"] == {
        "batch_size": 8,
        "data_dir": str(tmp_path / "data"),
    }
    assert calls["to_onnx"]["input_shapes"] == [("B", 28, 28, 1)]
    assert calls["to_onnx"]["input_params"] == {"deterministic": True}
    assert calls["save_model"] == str(tmp_path / "docs" / "mnist_dinov3_model.onnx")
    assert calls["visualize_results"]["output_dir"] == str(tmp_path / "output")
    assert calls["test_onnx_model"]["output_path"] == str(
        tmp_path / "docs" / "mnist_dinov3_model.onnx"
    )
    assert Path(tmp_path / "docs").exists()
    assert Path(tmp_path / "docs" / "mnist_dinov3_model_config.json").exists()


def test_mnist_dinov3_run_onnx_uses_dino_default_path(monkeypatch):
    captured = {}

    def fake_main(*, args=None, description, default_onnx_model):
        captured["args"] = args
        captured["description"] = description
        captured["default_onnx_model"] = default_onnx_model

    monkeypatch.setattr(mnist_dinov3_run_onnx, "run_mnist_onnx_main", fake_main)

    mnist_dinov3_run_onnx.main()

    assert captured["args"] is None
    assert captured["description"] == "Test ONNX MNIST DINOv3 model"
    assert captured["default_onnx_model"] == "output/mnist_dinov3_model.onnx"
