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

    def validate(self) -> None:
        _require(self.max_translation >= 0.0, "max_translation must be >= 0.")
        _require(self.scale_min_x > 0.0, "scale_min_x must be > 0.")
        _require(self.scale_max_x > 0.0, "scale_max_x must be > 0.")
        _require(self.scale_min_y > 0.0, "scale_min_y must be > 0.")
        _require(self.scale_max_y > 0.0, "scale_max_y must be > 0.")
        _require(self.scale_min_x <= self.scale_max_x, "scale_min_x must be <= scale_max_x.")
        _require(self.scale_min_y <= self.scale_max_y, "scale_min_y must be <= scale_max_y.")
        _require(self.max_rotation >= 0.0, "max_rotation must be >= 0.")
        _require(self.elastic_alpha >= 0.0, "elastic_alpha must be >= 0.")
        _require(self.elastic_sigma > 0.0, "elastic_sigma must be > 0.")
        _require(self.rect_erase_height > 0, "rect_erase_height must be > 0.")
        _require(self.rect_erase_width > 0, "rect_erase_width must be > 0.")


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

    def validate(self) -> None:
        _require(self.batch_size > 0, "batch_size must be > 0.")
        _require(self.base_learning_rate > 0.0, "base_learning_rate must be > 0.")
        _require(
            self.num_epochs_to_train_now >= 0,
            "num_epochs_to_train_now must be >= 0.",
        )
        _require(self.warmup_epochs >= 0, "warmup_epochs must be >= 0.")
        _require(self.start_epoch >= 0, "start_epoch must be >= 0.")
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
class MnistDinoV3ModelConfig(ConfigMixin):
    img_size: int
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    num_classes: int
    num_storage_tokens: int = 0

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
