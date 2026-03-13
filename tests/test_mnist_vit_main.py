from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from jaxamples import mnist_vit


def get_dummy_dataloaders(batch_size: int):
    images = torch.ones(8, 1, 28, 28, dtype=torch.float32)
    labels = torch.arange(8, dtype=torch.int64) % 10
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, dataloader


def test_mnist_vit_main_applies_cli_overrides(monkeypatch, tmp_path):
    calls = {}
    train_dataloader, test_dataloader = get_dummy_dataloaders(batch_size=8)

    config = mnist_vit.get_default_config()
    config.training.num_epochs_to_train_now = 1
    config.training.data_dir = str(tmp_path / "data")
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")
    config.training.output_dir = str(tmp_path / "output")
    config.onnx.output_path = str(tmp_path / "docs" / "mnist_vit_model.onnx")

    monkeypatch.setattr(mnist_vit, "get_default_config", lambda: config)
    def fake_get_dataset_torch_dataloaders(batch_size, data_dir):
        calls["loader_args"] = {"batch_size": batch_size, "data_dir": data_dir}
        return train_dataloader, test_dataloader

    monkeypatch.setattr(
        mnist_vit,
        "get_dataset_torch_dataloaders",
        fake_get_dataset_torch_dataloaders,
    )
    monkeypatch.setattr(mnist_vit, "resolve_checkpoint_resume", lambda ckpt_dir: (None, 0))

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
        calls["save_model"] = output_path

    def fake_test_onnx_model(output_path, test_loader):
        calls["test_onnx_model"] = {
            "output_path": output_path,
            "test_batches": len(test_loader),
        }

    monkeypatch.setattr(mnist_vit, "train_model", fake_train_model)
    monkeypatch.setattr(mnist_vit, "visualize_results", fake_visualize_results)
    monkeypatch.setattr(mnist_vit, "to_onnx", fake_to_onnx)
    monkeypatch.setattr(mnist_vit.onnx, "save_model", fake_save_model)
    monkeypatch.setattr(mnist_vit, "allclose", lambda *args, **kwargs: True)
    monkeypatch.setattr(mnist_vit, "test_onnx_model", fake_test_onnx_model)

    mnist_vit.main(
        [
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--onnx-output",
            str(tmp_path / "docs" / "mnist_vit_model.onnx"),
        ]
    )

    assert calls["train_model"]["start_epoch"] == 0
    assert calls["loader_args"] == {
        "batch_size": 8,
        "data_dir": str(tmp_path / "data"),
    }
    assert calls["to_onnx"]["input_shapes"] == [("B", 28, 28, 1)]
    assert calls["to_onnx"]["input_params"] == {"deterministic": True}
    assert calls["save_model"] == str(tmp_path / "docs" / "mnist_vit_model.onnx")
    assert calls["visualize_results"]["output_dir"] == str(tmp_path / "output")
    assert calls["test_onnx_model"]["output_path"] == str(
        tmp_path / "docs" / "mnist_vit_model.onnx"
    )
    assert Path(tmp_path / "docs").exists()
    assert Path(tmp_path / "docs" / "mnist_vit_model_config.json").exists()
