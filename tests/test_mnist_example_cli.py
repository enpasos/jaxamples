import json
import os
from pathlib import Path

from jaxamples.mnist_config import (
    AugmentationConfig,
    MnistExampleConfig,
    MnistVitModelConfig,
    OnnxConfig,
    TrainingConfig,
)
from jaxamples.mnist_example_cli import apply_common_overrides, parse_example_args


def test_parse_example_args_supports_runtime_overrides():
    args = parse_example_args(
        [
            "--epochs",
            "3",
            "--batch-size",
            "16",
            "--data-dir",
            "./mnist-data",
            "--checkpoint-dir",
            "./checkpoints",
            "--onnx-output",
            "output/example.onnx",
            "--output-dir",
            "./artifacts",
            "--seed",
            "42",
            "--skip-training",
        ],
        description="test",
        default_onnx_output="output/default.onnx",
    )

    assert args.epochs == 3
    assert args.batch_size == 16
    assert args.data_dir == "./mnist-data"
    assert args.checkpoint_dir == "./checkpoints"
    assert args.onnx_output == "output/example.onnx"
    assert args.output_dir == "./artifacts"
    assert args.seed == 42
    assert args.skip_training is True


def test_apply_common_overrides_updates_config_paths_and_flags(tmp_path):
    config = MnistExampleConfig(
        seed=1,
        training=TrainingConfig(
            enable_training=True,
            batch_size=64,
            base_learning_rate=0.001,
            num_epochs_to_train_now=10,
            warmup_epochs=1,
            checkpoint_dir="./default-checkpoints",
            data_dir="./data",
            augmentation=AugmentationConfig(
                enable_translation=True,
                max_translation=1.0,
                enable_scaling=False,
                scale_min_x=1.0,
                scale_max_x=1.0,
                scale_min_y=1.0,
                scale_max_y=1.0,
                enable_rotation=False,
                max_rotation=0.0,
                enable_elastic=False,
                elastic_alpha=0.0,
                elastic_sigma=1.0,
                enable_rect_erasing=False,
                rect_erase_height=1,
                rect_erase_width=1,
            ),
        ),
        model=MnistVitModelConfig(
            height=28,
            width=28,
            num_hiddens=64,
            num_layers=2,
            num_heads=4,
            mlp_dim=128,
            num_classes=10,
            embed_dims=[16, 32, 64],
            kernel_size=3,
            strides=[1, 2, 2],
            embedding_type="conv",
            embedding_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            mlp_dropout_rate=0.1,
        ),
        onnx=OnnxConfig(
            model_name="example",
            output_path="output/default.onnx",
            input_shapes=[("B", 28, 28, 1)],
            input_params={"deterministic": True},
        ),
    )
    args = parse_example_args(
        [
            "--epochs",
            "5",
            "--batch-size",
            "8",
            "--data-dir",
            str(tmp_path / "data"),
            "--checkpoint-dir",
            str(tmp_path / "ckpts"),
            "--onnx-output",
            str(tmp_path / "output" / "model.onnx"),
            "--seed",
            "99",
            "--skip-training",
        ],
        description="test",
        default_onnx_output="output/default.onnx",
    )

    updated = apply_common_overrides(config, args)

    assert updated.seed == 99
    assert updated.training.num_epochs_to_train_now == 5
    assert updated.training.batch_size == 8
    assert updated.training.data_dir == str(tmp_path / "data")
    assert updated.training.checkpoint_dir == os.path.abspath(
        str(tmp_path / "ckpts")
    )
    assert updated.onnx.output_path == str(tmp_path / "output" / "model.onnx")
    assert updated.training.enable_training is False


def test_apply_common_overrides_redirects_default_onnx_output_into_output_dir(tmp_path):
    config = MnistExampleConfig(
        seed=1,
        training=TrainingConfig(
            enable_training=True,
            batch_size=64,
            base_learning_rate=0.001,
            num_epochs_to_train_now=10,
            warmup_epochs=1,
            checkpoint_dir="./default-checkpoints",
            data_dir="./data",
            augmentation=AugmentationConfig(
                enable_translation=True,
                max_translation=1.0,
                enable_scaling=False,
                scale_min_x=1.0,
                scale_max_x=1.0,
                scale_min_y=1.0,
                scale_max_y=1.0,
                enable_rotation=False,
                max_rotation=0.0,
                enable_elastic=False,
                elastic_alpha=0.0,
                elastic_sigma=1.0,
                enable_rect_erasing=False,
                rect_erase_height=1,
                rect_erase_width=1,
            ),
        ),
        model=MnistVitModelConfig(
            height=28,
            width=28,
            num_hiddens=64,
            num_layers=2,
            num_heads=4,
            mlp_dim=128,
            num_classes=10,
            embed_dims=[16, 32, 64],
            kernel_size=3,
            strides=[1, 2, 2],
            embedding_type="conv",
            embedding_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            mlp_dropout_rate=0.1,
        ),
        onnx=OnnxConfig(
            model_name="example",
            output_path="output/default.onnx",
            input_shapes=[("B", 28, 28, 1)],
            input_params={"deterministic": True},
        ),
    )
    args = parse_example_args(
        [
            "--output-dir",
            str(tmp_path / "artifacts"),
        ],
        description="test",
        default_onnx_output="output/default.onnx",
    )

    updated = apply_common_overrides(config, args)

    assert updated.training.output_dir == os.path.abspath(str(tmp_path / "artifacts"))
    assert updated.onnx.output_path == str(tmp_path / "artifacts" / "default.onnx")
    assert updated.config_output_path() == tmp_path / "artifacts" / "default_config.json"


def test_apply_common_overrides_rejects_invalid_batch_size():
    config = MnistExampleConfig(
        seed=1,
        training=TrainingConfig(
            enable_training=True,
            batch_size=64,
            base_learning_rate=0.001,
            num_epochs_to_train_now=10,
            warmup_epochs=1,
            checkpoint_dir="./default-checkpoints",
            data_dir="./data",
            augmentation=AugmentationConfig(
                enable_translation=True,
                max_translation=1.0,
                enable_scaling=False,
                scale_min_x=1.0,
                scale_max_x=1.0,
                scale_min_y=1.0,
                scale_max_y=1.0,
                enable_rotation=False,
                max_rotation=0.0,
                enable_elastic=False,
                elastic_alpha=0.0,
                elastic_sigma=1.0,
                enable_rect_erasing=False,
                rect_erase_height=1,
                rect_erase_width=1,
            ),
        ),
        model=MnistVitModelConfig(
            height=28,
            width=28,
            num_hiddens=64,
            num_layers=2,
            num_heads=4,
            mlp_dim=128,
            num_classes=10,
            embed_dims=[16, 32, 64],
            kernel_size=3,
            strides=[1, 2, 2],
            embedding_type="conv",
            embedding_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            mlp_dropout_rate=0.1,
        ),
        onnx=OnnxConfig(
            model_name="example",
            output_path="output/default.onnx",
            input_shapes=[("B", 28, 28, 1)],
            input_params={"deterministic": True},
        ),
    )
    args = parse_example_args(
        ["--batch-size", "0"],
        description="test",
        default_onnx_output="output/default.onnx",
    )

    try:
        apply_common_overrides(config, args)
    except ValueError as exc:
        assert "batch_size must be > 0" in str(exc)
    else:
        raise AssertionError("Expected invalid batch size to raise ValueError")


def test_mnist_example_config_writes_json_snapshot(tmp_path):
    config = MnistExampleConfig(
        seed=1,
        training=TrainingConfig(
            enable_training=True,
            batch_size=64,
            base_learning_rate=0.001,
            num_epochs_to_train_now=10,
            warmup_epochs=1,
            checkpoint_dir="./default-checkpoints",
            data_dir="./data",
            augmentation=AugmentationConfig(
                enable_translation=True,
                max_translation=1.0,
                enable_scaling=False,
                scale_min_x=1.0,
                scale_max_x=1.0,
                scale_min_y=1.0,
                scale_max_y=1.0,
                enable_rotation=False,
                max_rotation=0.0,
                enable_elastic=False,
                elastic_alpha=0.0,
                elastic_sigma=1.0,
                enable_rect_erasing=False,
                rect_erase_height=1,
                rect_erase_width=1,
            ),
        ),
        model=MnistVitModelConfig(
            height=28,
            width=28,
            num_hiddens=64,
            num_layers=2,
            num_heads=4,
            mlp_dim=128,
            num_classes=10,
            embed_dims=[16, 32, 64],
            kernel_size=3,
            strides=[1, 2, 2],
            embedding_type="conv",
            embedding_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            mlp_dropout_rate=0.1,
        ),
        onnx=OnnxConfig(
            model_name="example",
            output_path=str(tmp_path / "docs" / "default.onnx"),
            input_shapes=[("B", 28, 28, 1)],
            input_params={"deterministic": True},
        ),
    )

    written_path = config.write_json(config.config_output_path())

    assert written_path == Path(tmp_path / "docs" / "default_config.json")
    loaded = json.loads(written_path.read_text(encoding="utf-8"))
    assert loaded["seed"] == 1
    assert loaded["onnx"]["output_path"] == str(tmp_path / "docs" / "default.onnx")
