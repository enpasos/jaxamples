import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ConfigMixin:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, output_path: str | Path) -> Path:
        resolved_path = Path(output_path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return resolved_path

    def validate(self) -> None:
        return None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_dropout(rate: float, name: str) -> None:
    _require(0.0 <= rate < 1.0, f"{name} must be in [0.0, 1.0).")


def _validate_probability(rate: float, name: str) -> None:
    _require(0.0 <= rate <= 1.0, f"{name} must be in [0.0, 1.0].")


@dataclass
class AugmentationConfig(ConfigMixin):
    enable_translation: bool
    max_translation: float
    enable_scaling: bool
    scale_min_x: float
    scale_max_x: float
    scale_min_y: float
    scale_max_y: float
    enable_rotation: bool
    max_rotation: float
    enable_elastic: bool
    elastic_alpha: float
    elastic_sigma: float
    enable_rect_erasing: bool
    rect_erase_height: int
    rect_erase_width: int
    translation_probability: float = 1.0
    scaling_probability: float = 1.0
    rotation_probability: float = 1.0
    elastic_probability: float = 1.0
    rect_erasing_probability: float = 1.0

    def validate(self) -> None:
        _require(self.max_translation >= 0.0, "max_translation must be >= 0.")
        _validate_probability(self.translation_probability, "translation_probability")
        _require(self.scale_min_x > 0.0, "scale_min_x must be > 0.")
        _require(self.scale_max_x > 0.0, "scale_max_x must be > 0.")
        _require(self.scale_min_y > 0.0, "scale_min_y must be > 0.")
        _require(self.scale_max_y > 0.0, "scale_max_y must be > 0.")
        _require(self.scale_min_x <= self.scale_max_x, "scale_min_x must be <= scale_max_x.")
        _require(self.scale_min_y <= self.scale_max_y, "scale_min_y must be <= scale_max_y.")
        _validate_probability(self.scaling_probability, "scaling_probability")
        _require(self.max_rotation >= 0.0, "max_rotation must be >= 0.")
        _validate_probability(self.rotation_probability, "rotation_probability")
        _require(self.elastic_alpha >= 0.0, "elastic_alpha must be >= 0.")
        _require(self.elastic_sigma > 0.0, "elastic_sigma must be > 0.")
        _validate_probability(self.elastic_probability, "elastic_probability")
        _require(self.rect_erase_height > 0, "rect_erase_height must be > 0.")
        _require(self.rect_erase_width > 0, "rect_erase_width must be > 0.")
        _validate_probability(self.rect_erasing_probability, "rect_erasing_probability")


@dataclass
class TrainingConfig(ConfigMixin):
    enable_training: bool
    batch_size: int
    base_learning_rate: float
    num_epochs_to_train_now: int
    warmup_epochs: int
    checkpoint_dir: str
    data_dir: str
    augmentation: AugmentationConfig
    start_epoch: int = 0
    output_dir: str | None = "output"
    weight_decay: float = 1e-4

    def validate(self) -> None:
        _require(self.batch_size > 0, "batch_size must be > 0.")
        _require(self.base_learning_rate > 0.0, "base_learning_rate must be > 0.")
        _require(
            self.num_epochs_to_train_now >= 0,
            "num_epochs_to_train_now must be >= 0.",
        )
        _require(self.warmup_epochs >= 0, "warmup_epochs must be >= 0.")
        _require(self.start_epoch >= 0, "start_epoch must be >= 0.")
        _require(self.weight_decay >= 0.0, "weight_decay must be >= 0.")
        _require(bool(self.checkpoint_dir), "checkpoint_dir must not be empty.")
        _require(bool(self.data_dir), "data_dir must not be empty.")
        if self.output_dir is not None:
            _require(bool(self.output_dir), "output_dir must not be empty.")
        self.augmentation.validate()


@dataclass
class OnnxConfig(ConfigMixin):
    model_name: str
    output_path: str
    input_shapes: list[tuple[Any, ...]]
    input_params: dict[str, Any]

    def validate(self) -> None:
        _require(bool(self.model_name), "model_name must not be empty.")
        _require(bool(self.output_path), "output_path must not be empty.")
        _require(bool(self.input_shapes), "input_shapes must not be empty.")
        for shape in self.input_shapes:
            _require(len(shape) == 4, f"input shape {shape} must be 4D.")


@dataclass
class MnistVitModelConfig(ConfigMixin):
    height: int
    width: int
    num_hiddens: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    num_classes: int
    embed_dims: list[int]
    kernel_size: int
    strides: list[int]
    embedding_type: str
    embedding_dropout_rate: float
    attention_dropout_rate: float
    mlp_dropout_rate: float

    def validate(self) -> None:
        _require(self.height > 0, "height must be > 0.")
        _require(self.width > 0, "width must be > 0.")
        _require(self.num_hiddens > 0, "num_hiddens must be > 0.")
        _require(self.num_layers > 0, "num_layers must be > 0.")
        _require(self.num_heads > 0, "num_heads must be > 0.")
        _require(self.mlp_dim > 0, "mlp_dim must be > 0.")
        _require(self.num_classes >= 2, "num_classes must be >= 2.")
        _require(bool(self.embed_dims), "embed_dims must not be empty.")
        _require(all(dim > 0 for dim in self.embed_dims), "embed_dims must all be > 0.")
        _require(self.kernel_size > 0, "kernel_size must be > 0.")
        _require(bool(self.strides), "strides must not be empty.")
        _require(all(step > 0 for step in self.strides), "strides must all be > 0.")
        _require(
            len(self.embed_dims) == len(self.strides),
            "embed_dims and strides must have the same length.",
        )
        _require(
            self.embedding_type in {"conv", "patch"},
            "embedding_type must be 'conv' or 'patch'.",
        )
        _validate_dropout(self.embedding_dropout_rate, "embedding_dropout_rate")
        _validate_dropout(self.attention_dropout_rate, "attention_dropout_rate")
        _validate_dropout(self.mlp_dropout_rate, "mlp_dropout_rate")


@dataclass
class MnistCnnModelConfig(ConfigMixin):
    height: int
    width: int
    num_classes: int
    conv_channels: list[int]
    dense_hidden_dim: int
    feature_dropout_rate: float = 0.1
    classifier_dropout_rate: float = 0.3

    def validate(self) -> None:
        _require(self.height > 0, "height must be > 0.")
        _require(self.width > 0, "width must be > 0.")
        _require(self.num_classes >= 2, "num_classes must be >= 2.")
        _require(
            len(self.conv_channels) == 4,
            "conv_channels must contain exactly four entries.",
        )
        _require(
            all(channel > 0 for channel in self.conv_channels),
            "conv_channels must all be > 0.",
        )
        _require(self.dense_hidden_dim > 0, "dense_hidden_dim must be > 0.")
        _validate_dropout(self.feature_dropout_rate, "feature_dropout_rate")
        _validate_dropout(self.classifier_dropout_rate, "classifier_dropout_rate")


@dataclass
class MnistDinoV3ModelConfig(ConfigMixin):
    img_size: int
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    num_classes: int
    num_storage_tokens: int = 0
    head_hidden_dim: int = 192
    head_dropout_rate: float = 0.1
    pool_features: str = "cls_mean"

    def validate(self) -> None:
        _require(self.img_size > 0, "img_size must be > 0.")
        _require(self.patch_size > 0, "patch_size must be > 0.")
        _require(
            self.img_size % self.patch_size == 0,
            "img_size must be divisible by patch_size.",
        )
        _require(self.embed_dim > 0, "embed_dim must be > 0.")
        _require(self.depth > 0, "depth must be > 0.")
        _require(self.num_heads > 0, "num_heads must be > 0.")
        _require(
            self.embed_dim % self.num_heads == 0,
            "embed_dim must be divisible by num_heads.",
        )
        _require(self.num_classes >= 2, "num_classes must be >= 2.")
        _require(self.num_storage_tokens >= 0, "num_storage_tokens must be >= 0.")
        _require(self.head_hidden_dim > 0, "head_hidden_dim must be > 0.")
        _validate_dropout(self.head_dropout_rate, "head_dropout_rate")
        _require(
            self.pool_features in {"cls", "cls_mean"},
            "pool_features must be 'cls' or 'cls_mean'.",
        )


@dataclass
class MnistExampleConfig(ConfigMixin):
    seed: int
    training: TrainingConfig
    model: Any
    onnx: OnnxConfig

    def validate(self) -> None:
        _require(self.seed >= 0, "seed must be >= 0.")
        self.training.validate()
        self.onnx.validate()
        if hasattr(self.model, "validate"):
            self.model.validate()

        first_input_shape = self.onnx.input_shapes[0]
        _require(first_input_shape[0] == "B", "first ONNX input dim must be 'B'.")
        _require(first_input_shape[-1] == 1, "MNIST ONNX input must use a single channel.")

        if isinstance(self.model, MnistVitModelConfig):
            _require(
                first_input_shape[1:3] == (self.model.height, self.model.width),
                "ONNX input shape must match the ViT image size.",
            )
        if isinstance(self.model, MnistCnnModelConfig):
            _require(
                first_input_shape[1:3] == (self.model.height, self.model.width),
                "ONNX input shape must match the CNN image size.",
            )
        if isinstance(self.model, MnistDinoV3ModelConfig):
            _require(
                first_input_shape[1:3] == (self.model.img_size, self.model.img_size),
                "ONNX input shape must match the DINOv3 image size.",
            )

    def artifact_dir(self) -> Path:
        if self.training.output_dir:
            return Path(self.training.output_dir)
        return Path("output")

    def metrics_csv_path(self) -> Path:
        return self.artifact_dir() / "test_accuracy_metrics.csv"

    def metrics_plot_path(self) -> Path:
        return self.artifact_dir() / "test_accuracy_metrics.png"

    def config_output_path(self) -> Path:
        onnx_path = Path(self.onnx.output_path)
        return onnx_path.with_name(f"{onnx_path.stem}_config.json")

    def benchmark_memory_path(self) -> Path:
        return self.artifact_dir() / "benchmark_memory.jsonl"


def shared_mnist_augmentation_config() -> AugmentationConfig:
    return AugmentationConfig(
        enable_translation=True,
        max_translation=2.5,
        translation_probability=0.8,
        enable_scaling=True,
        scale_min_x=0.9,
        scale_max_x=1.1,
        scale_min_y=0.9,
        scale_max_y=1.1,
        scaling_probability=0.7,
        enable_rotation=True,
        max_rotation=10.0,
        rotation_probability=0.7,
        enable_elastic=True,
        elastic_alpha=1.2,
        elastic_sigma=0.7,
        elastic_probability=0.35,
        enable_rect_erasing=False,
        rect_erase_height=2,
        rect_erase_width=20,
        rect_erasing_probability=0.0,
    )


def shared_mnist_training_config(
    *,
    checkpoint_dir: str,
    output_dir: str,
    data_dir: str = "./data",
    enable_training: bool = True,
    batch_size: int = 64,
    base_learning_rate: float = 1e-4,
    num_epochs_to_train_now: int = 500,
    warmup_epochs: int = 5,
    weight_decay: float = 1e-4,
) -> TrainingConfig:
    return TrainingConfig(
        enable_training=enable_training,
        batch_size=batch_size,
        base_learning_rate=base_learning_rate,
        num_epochs_to_train_now=num_epochs_to_train_now,
        warmup_epochs=warmup_epochs,
        checkpoint_dir=checkpoint_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        weight_decay=weight_decay,
        augmentation=shared_mnist_augmentation_config(),
    )
