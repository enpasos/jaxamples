import argparse
import csv
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from flax import nnx, serialization
from jax2onnx import allclose, to_onnx

from jaxamples import mnist_training as mnist_training_lib
from jaxamples import mnist_vit as mnist_vit_lib
from jaxamples.mnist_benchmark_memory import build_benchmark_record
from jaxamples.mnist_config import (
    AugmentationConfig,
    ConfigMixin,
    MnistVitModelConfig,
    OnnxConfig,
    TrainingConfig,
)
from jaxamples.mnist_example_cli import apply_common_overrides


VisionTransformer = mnist_vit_lib.VisionTransformer
get_dataset_torch_dataloaders = mnist_vit_lib.get_dataset_torch_dataloaders
jax_collate = mnist_vit_lib.jax_collate
AugmentationParams = mnist_vit_lib.AugmentationParams
augment_data_batch = mnist_vit_lib.augment_data_batch
visualize_augmented_images = mnist_vit_lib.visualize_augmented_images
train_step = mnist_vit_lib.train_step
eval_step = mnist_vit_lib.eval_step
pred_step = mnist_vit_lib.pred_step
visualize_incorrect_classifications = mnist_vit_lib.visualize_incorrect_classifications
lr_schedule = mnist_vit_lib.lr_schedule
create_optimizer = mnist_vit_lib.create_optimizer
compute_mean_and_spread = mnist_vit_lib.compute_mean_and_spread
load_and_plot_test_accuracy_metrics = mnist_training_lib.load_and_plot_test_accuracy_metrics
save_model = mnist_training_lib.save_model
load_model = mnist_training_lib.load_model
resolve_checkpoint_resume = mnist_training_lib.resolve_checkpoint_resume
test_onnx_model = mnist_training_lib.test_onnx_model
visualize_results = mnist_vit_lib.visualize_results


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


_SEARCH_TOLERANCE = 1e-12
_CURRENT_SEARCH_VERSION = 4
_ARTIFACT_REFRESH_INTERVAL = 5
_NON_MONOTONIC_GRID_POINTS = {
    "elastic_sigma": 41,
}
_SEARCH_PARAMETER_ORDER = (
    "translation_probability",
    "max_translation",
    "scaling_probability",
    "scale_span_x",
    "scale_span_y",
    "rotation_probability",
    "max_rotation",
    "elastic_probability",
    "elastic_alpha",
    "elastic_sigma",
    "rect_erasing_probability",
    "rect_erase_height",
    "rect_erase_width",
)


@dataclass
class AugmentationSearchConfig(ConfigMixin):
    anchor_max_examples: int = 2048
    anchor_examples_per_class: int = 256
    anchor_min_margin: float = 0.5
    invariance_threshold: float = 0.995
    margin_retention_threshold: float = 0.9
    candidate_samples: int = 3
    strength_step: float = 0.2
    min_strength_step: float = 0.05
    search_every_n_epochs: int = 2
    annealed_strength_step: float = 0.05
    annealed_min_strength_step: float = 0.01
    train_plateau_window: int = 10
    train_plateau_patience: int = 30
    train_plateau_min_improvement: float = 1e-4
    anneal_after_plateau_fraction: float = 0.5
    max_strength: float = 1.5

    def validate(self) -> None:
        _require(
            self.anchor_max_examples > 0,
            "anchor_max_examples must be > 0.",
        )
        _require(
            self.anchor_examples_per_class > 0,
            "anchor_examples_per_class must be > 0.",
        )
        _require(
            self.anchor_min_margin >= 0.0,
            "anchor_min_margin must be >= 0.",
        )
        _require(
            0.0 < self.invariance_threshold <= 1.0,
            "invariance_threshold must be in (0.0, 1.0].",
        )
        _require(
            self.margin_retention_threshold > 0.0,
            "margin_retention_threshold must be > 0.",
        )
        _require(
            self.candidate_samples > 0,
            "candidate_samples must be > 0.",
        )
        _require(self.strength_step > 0.0, "strength_step must be > 0.")
        _require(
            self.min_strength_step > 0.0,
            "min_strength_step must be > 0.",
        )
        _require(
            self.search_every_n_epochs > 0,
            "search_every_n_epochs must be > 0.",
        )
        _require(
            self.max_strength >= 0.0,
            "max_strength must be >= 0.",
        )
        _require(
            self.min_strength_step <= self.strength_step,
            "min_strength_step must be <= strength_step.",
        )
        _require(
            self.min_strength_step <= self.max_strength,
            "min_strength_step must be <= max_strength.",
        )
        _require(
            self.annealed_strength_step > 0.0,
            "annealed_strength_step must be > 0.",
        )
        _require(
            self.annealed_min_strength_step > 0.0,
            "annealed_min_strength_step must be > 0.",
        )
        _require(
            self.annealed_strength_step <= self.strength_step,
            "annealed_strength_step must be <= strength_step.",
        )
        _require(
            self.annealed_min_strength_step <= self.min_strength_step,
            "annealed_min_strength_step must be <= min_strength_step.",
        )
        _require(
            self.annealed_min_strength_step <= self.annealed_strength_step,
            "annealed_min_strength_step must be <= annealed_strength_step.",
        )
        _require(
            self.train_plateau_window > 0,
            "train_plateau_window must be > 0.",
        )
        _require(
            self.train_plateau_patience >= 0,
            "train_plateau_patience must be >= 0.",
        )
        _require(
            self.train_plateau_min_improvement >= 0.0,
            "train_plateau_min_improvement must be >= 0.",
        )
        _require(
            0.0 <= self.anneal_after_plateau_fraction <= 1.0,
            "anneal_after_plateau_fraction must be in [0.0, 1.0].",
        )


@dataclass
class AugmentationSearchState(ConfigMixin):
    search_version: int
    parameter_values: Dict[str, float]
    last_anchor_size: int
    last_anchor_num_classes: int
    last_anchor_min_class_count: int
    last_anchor_mean_margin: float
    last_retention: float
    last_margin_retention: float
    last_updated_parameters: list[str]
    last_event: str
    best_train_accuracy_metric: float
    epochs_since_train_improvement: int
    search_frozen: bool
    frozen_epoch: int | None

    def validate(self) -> None:
        _require(
            self.search_version == _CURRENT_SEARCH_VERSION,
            "search_version does not match the current search implementation.",
        )
        _require(
            set(self.parameter_values) == set(_SEARCH_PARAMETER_ORDER),
            "parameter_values must contain exactly the configured search parameters.",
        )
        for name, value in self.parameter_values.items():
            _require(0.0 <= value <= 1.0, f"{name} must be in [0.0, 1.0].")
        _require(self.last_anchor_size >= 0, "last_anchor_size must be >= 0.")
        _require(
            self.last_anchor_num_classes >= 0,
            "last_anchor_num_classes must be >= 0.",
        )
        _require(
            self.last_anchor_min_class_count >= 0,
            "last_anchor_min_class_count must be >= 0.",
        )
        _require(
            self.last_anchor_mean_margin >= 0.0,
            "last_anchor_mean_margin must be >= 0.",
        )
        _require(0.0 <= self.last_retention <= 1.0, "last_retention must be in [0.0, 1.0].")
        _require(
            self.last_margin_retention >= 0.0,
            "last_margin_retention must be >= 0.",
        )
        _require(
            0.0 <= self.best_train_accuracy_metric <= 1.0,
            "best_train_accuracy_metric must be in [0.0, 1.0].",
        )
        _require(
            self.epochs_since_train_improvement >= 0,
            "epochs_since_train_improvement must be >= 0.",
        )
        if self.frozen_epoch is not None:
            _require(self.frozen_epoch >= 0, "frozen_epoch must be >= 0.")


@dataclass
class AnchorSet:
    images: jax.Array
    labels: jax.Array
    class_counts: list[int]
    mean_margin: float


@dataclass
class ParameterSearchUpdate:
    parameter: str
    old_value: float
    new_value: float
    retention: float
    mean_margin: float
    margin_retention: float


@dataclass
class CandidateEvaluation:
    retention: float
    mean_margin: float
    margin_retention: float


def _is_acceptable_candidate(
    candidate: CandidateEvaluation,
    search_config: AugmentationSearchConfig,
) -> bool:
    return (
        candidate.retention >= search_config.invariance_threshold - _SEARCH_TOLERANCE
        and candidate.margin_retention
        >= search_config.margin_retention_threshold - _SEARCH_TOLERANCE
    )


def _lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def _effective_search_config(
    search_config: AugmentationSearchConfig,
    state: AugmentationSearchState,
) -> AugmentationSearchConfig:
    if state.search_frozen or search_config.train_plateau_patience <= 0:
        return search_config

    stale_fraction = min(
        1.0,
        state.epochs_since_train_improvement
        / max(1, search_config.train_plateau_patience),
    )
    if stale_fraction <= search_config.anneal_after_plateau_fraction + _SEARCH_TOLERANCE:
        return search_config

    if search_config.anneal_after_plateau_fraction >= 1.0 - _SEARCH_TOLERANCE:
        anneal_progress = 1.0
    else:
        anneal_progress = (
            stale_fraction - search_config.anneal_after_plateau_fraction
        ) / (1.0 - search_config.anneal_after_plateau_fraction)
    anneal_progress = min(1.0, max(0.0, anneal_progress))

    effective_strength_step = _lerp(
        search_config.strength_step,
        search_config.annealed_strength_step,
        anneal_progress,
    )
    effective_min_strength_step = _lerp(
        search_config.min_strength_step,
        search_config.annealed_min_strength_step,
        anneal_progress,
    )
    effective_strength_step = max(effective_strength_step, effective_min_strength_step)
    return replace(
        search_config,
        strength_step=effective_strength_step,
        min_strength_step=effective_min_strength_step,
    )


def _is_scheduled_search_epoch(
    epoch: int,
    search_config: AugmentationSearchConfig,
) -> bool:
    return (epoch - 1) % search_config.search_every_n_epochs == 0


def _should_run_search_epoch(
    epoch: int,
    state: AugmentationSearchState,
    search_config: AugmentationSearchConfig,
) -> bool:
    if state.search_frozen:
        return True
    if state.last_event in {"init", "reset_search_state"}:
        return True
    return _is_scheduled_search_epoch(epoch, search_config)


def _should_refresh_artifacts(
    epoch: int,
    *,
    final_epoch: int,
    refresh_interval: int = _ARTIFACT_REFRESH_INTERVAL,
) -> bool:
    if refresh_interval <= 1:
        return True
    return epoch >= final_epoch or epoch % refresh_interval == 0


def _train_plateau_metric(
    train_accuracies: Sequence[float],
    search_config: AugmentationSearchConfig,
) -> float | None:
    if not train_accuracies:
        return None
    if len(train_accuracies) < search_config.train_plateau_window:
        return float(train_accuracies[-1])
    return float(np.mean(train_accuracies[-search_config.train_plateau_window :]))


def update_search_train_plateau_state(
    state: AugmentationSearchState,
    train_accuracies: Sequence[float],
    epoch: int,
    search_config: AugmentationSearchConfig,
) -> tuple[AugmentationSearchState, float | None, str | None]:
    metric = _train_plateau_metric(train_accuracies, search_config)
    if metric is None:
        return state, None, None

    best_train_accuracy_metric = state.best_train_accuracy_metric
    epochs_since_train_improvement = state.epochs_since_train_improvement
    search_frozen = state.search_frozen
    frozen_epoch = state.frozen_epoch
    last_event = state.last_event
    freeze_message = None

    if (
        best_train_accuracy_metric <= _SEARCH_TOLERANCE
        or metric
        > best_train_accuracy_metric + search_config.train_plateau_min_improvement
    ):
        best_train_accuracy_metric = metric
        epochs_since_train_improvement = 0
    elif (
        len(train_accuracies) >= search_config.train_plateau_window
        and search_config.train_plateau_patience > 0
        and not search_frozen
    ):
        epochs_since_train_improvement += 1
        if epochs_since_train_improvement >= search_config.train_plateau_patience:
            search_frozen = True
            frozen_epoch = epoch
            last_event = "search_train_plateau_frozen"
            freeze_message = (
                "[search] train plateau detected. Freezing augmentation search at "
                f"epoch {epoch} after {epochs_since_train_improvement} stale epochs; "
                f"best smoothed clean train accuracy: {best_train_accuracy_metric:.4f}"
            )

    return (
        replace(
            state,
            best_train_accuracy_metric=best_train_accuracy_metric,
            epochs_since_train_improvement=epochs_since_train_improvement,
            search_frozen=search_frozen,
            frozen_epoch=frozen_epoch,
            last_event=last_event,
        ),
        metric,
        freeze_message,
    )


@dataclass
class MnistVitAugSearchConfig(ConfigMixin):
    seed: int
    training: TrainingConfig
    model: MnistVitModelConfig
    onnx: OnnxConfig
    search: AugmentationSearchConfig

    def validate(self) -> None:
        _require(self.seed >= 0, "seed must be >= 0.")
        self.training.validate()
        self.model.validate()
        self.onnx.validate()
        self.search.validate()

        first_input_shape = self.onnx.input_shapes[0]
        _require(first_input_shape[0] == "B", "first ONNX input dim must be 'B'.")
        _require(first_input_shape[-1] == 1, "MNIST ONNX input must use a single channel.")
        _require(
            first_input_shape[1:3] == (self.model.height, self.model.width),
            "ONNX input shape must match the ViT image size.",
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


def default_search_state(search_config: AugmentationSearchConfig) -> AugmentationSearchState:
    del search_config
    return AugmentationSearchState(
        search_version=_CURRENT_SEARCH_VERSION,
        parameter_values={name: 0.0 for name in _SEARCH_PARAMETER_ORDER},
        last_anchor_size=0,
        last_anchor_num_classes=0,
        last_anchor_min_class_count=0,
        last_anchor_mean_margin=0.0,
        last_retention=1.0,
        last_margin_retention=1.0,
        last_updated_parameters=[],
        last_event="init",
        best_train_accuracy_metric=0.0,
        epochs_since_train_improvement=0,
        search_frozen=False,
        frozen_epoch=None,
    )


def _clamp_strength(value: float) -> float:
    return min(1.0, max(0.0, value))


def _parameter_max_value(
    name: str,
    reference: AugmentationConfig,
    search_config: AugmentationSearchConfig,
) -> float:
    multiplier = search_config.max_strength
    if name.endswith("probability"):
        return 1.0
    if name == "max_translation":
        return max(reference.max_translation * multiplier, 6.0)
    if name == "scale_span_x":
        return max(
            max(1.0 - reference.scale_min_x, reference.scale_max_x - 1.0) * multiplier,
            0.2,
        )
    if name == "scale_span_y":
        return max(
            max(1.0 - reference.scale_min_y, reference.scale_max_y - 1.0) * multiplier,
            0.2,
        )
    if name == "max_rotation":
        return max(reference.max_rotation * multiplier, 18.0)
    if name == "elastic_alpha":
        return max(reference.elastic_alpha * multiplier, 2.0)
    if name == "elastic_sigma":
        return max(reference.elastic_sigma * multiplier, 2.0)
    if name == "rect_erase_height":
        return float(max(int(round(reference.rect_erase_height * multiplier)), 4))
    if name == "rect_erase_width":
        return float(max(int(round(reference.rect_erase_width * multiplier)), 24))
    raise KeyError(f"Unsupported search parameter: {name}")


def _bootstrap_magnitude(
    name: str,
    reference: AugmentationConfig,
    search_config: AugmentationSearchConfig,
) -> float:
    max_value = _parameter_max_value(name, reference, search_config)
    if name == "max_translation":
        return min(max_value, 1.5)
    if name in {"scale_span_x", "scale_span_y"}:
        return min(max_value, 0.05)
    if name == "max_rotation":
        return min(max_value, 3.0)
    if name == "elastic_alpha":
        return min(max_value, 0.5)
    if name == "rect_erase_height":
        return min(max_value, 2.0)
    if name == "rect_erase_width":
        return min(max_value, 8.0)
    raise KeyError(f"Unsupported bootstrap parameter: {name}")


def _parameter_is_searchable(
    parameter_name: str,
    parameter_values: Dict[str, float],
) -> bool:
    if parameter_name == "max_translation":
        return parameter_values["translation_probability"] > _SEARCH_TOLERANCE
    if parameter_name in {"scale_span_x", "scale_span_y"}:
        return parameter_values["scaling_probability"] > _SEARCH_TOLERANCE
    if parameter_name == "max_rotation":
        return parameter_values["rotation_probability"] > _SEARCH_TOLERANCE
    if parameter_name == "elastic_alpha":
        return parameter_values["elastic_probability"] > _SEARCH_TOLERANCE
    if parameter_name == "elastic_sigma":
        return (
            parameter_values["elastic_probability"] > _SEARCH_TOLERANCE
            and parameter_values["elastic_alpha"] > _SEARCH_TOLERANCE
        )
    if parameter_name in {"rect_erase_height", "rect_erase_width"}:
        return parameter_values["rect_erasing_probability"] > _SEARCH_TOLERANCE
    return True


def build_parameter_scaled_augmentation(
    reference: AugmentationConfig,
    parameter_values: Dict[str, float],
    search_config: AugmentationSearchConfig,
    *,
    image_height: int | None = None,
    image_width: int | None = None,
) -> AugmentationConfig:
    values = {
        name: _clamp_strength(parameter_values.get(name, 0.0))
        for name in _SEARCH_PARAMETER_ORDER
    }

    translation_probability = values["translation_probability"]
    max_translation = values["max_translation"] * _parameter_max_value(
        "max_translation", reference, search_config
    )
    if translation_probability > _SEARCH_TOLERANCE and max_translation <= _SEARCH_TOLERANCE:
        max_translation = _bootstrap_magnitude("max_translation", reference, search_config)

    scaling_probability = values["scaling_probability"]
    scale_span_x = values["scale_span_x"] * _parameter_max_value(
        "scale_span_x", reference, search_config
    )
    scale_span_y = values["scale_span_y"] * _parameter_max_value(
        "scale_span_y", reference, search_config
    )
    if scaling_probability > _SEARCH_TOLERANCE and (
        scale_span_x <= _SEARCH_TOLERANCE and scale_span_y <= _SEARCH_TOLERANCE
    ):
        scale_span_x = _bootstrap_magnitude("scale_span_x", reference, search_config)
        scale_span_y = _bootstrap_magnitude("scale_span_y", reference, search_config)

    rotation_probability = values["rotation_probability"]
    max_rotation = values["max_rotation"] * _parameter_max_value(
        "max_rotation", reference, search_config
    )
    if rotation_probability > _SEARCH_TOLERANCE and max_rotation <= _SEARCH_TOLERANCE:
        max_rotation = _bootstrap_magnitude("max_rotation", reference, search_config)

    elastic_probability = values["elastic_probability"]
    elastic_alpha = values["elastic_alpha"] * _parameter_max_value(
        "elastic_alpha", reference, search_config
    )
    elastic_sigma = 1.0 + values["elastic_sigma"] * (
        _parameter_max_value("elastic_sigma", reference, search_config) - 1.0
    )
    if elastic_probability > _SEARCH_TOLERANCE and elastic_alpha <= _SEARCH_TOLERANCE:
        elastic_alpha = _bootstrap_magnitude("elastic_alpha", reference, search_config)

    rect_erasing_probability = values["rect_erasing_probability"]
    rect_erase_height = 1 + int(
        round(
            values["rect_erase_height"]
            * (_parameter_max_value("rect_erase_height", reference, search_config) - 1.0)
        )
    )
    rect_erase_width = 1 + int(
        round(
            values["rect_erase_width"]
            * (_parameter_max_value("rect_erase_width", reference, search_config) - 1.0)
        )
    )
    if rect_erasing_probability > _SEARCH_TOLERANCE and rect_erase_height <= 1:
        rect_erase_height = int(
            round(_bootstrap_magnitude("rect_erase_height", reference, search_config))
        )
    if rect_erasing_probability > _SEARCH_TOLERANCE and rect_erase_width <= 1:
        rect_erase_width = int(
            round(_bootstrap_magnitude("rect_erase_width", reference, search_config))
        )
    if image_height is not None:
        rect_erase_height = min(max(1, rect_erase_height), image_height)
    if image_width is not None:
        rect_erase_width = min(max(1, rect_erase_width), image_width)

    enable_translation = (
        translation_probability > _SEARCH_TOLERANCE and max_translation > _SEARCH_TOLERANCE
    )
    enable_scaling = scaling_probability > _SEARCH_TOLERANCE and (
        scale_span_x > _SEARCH_TOLERANCE or scale_span_y > _SEARCH_TOLERANCE
    )
    enable_rotation = (
        rotation_probability > _SEARCH_TOLERANCE and max_rotation > _SEARCH_TOLERANCE
    )
    enable_elastic = (
        elastic_probability > _SEARCH_TOLERANCE and elastic_alpha > _SEARCH_TOLERANCE
    )
    enable_rect_erasing = rect_erasing_probability > _SEARCH_TOLERANCE and (
        rect_erase_height > 1 or rect_erase_width > 1
    )

    return AugmentationConfig(
        enable_translation=enable_translation,
        max_translation=max_translation,
        translation_probability=translation_probability,
        enable_scaling=enable_scaling,
        scale_min_x=max(0.1, 1.0 - scale_span_x),
        scale_max_x=1.0 + scale_span_x,
        scale_min_y=max(0.1, 1.0 - scale_span_y),
        scale_max_y=1.0 + scale_span_y,
        scaling_probability=scaling_probability,
        enable_rotation=enable_rotation,
        max_rotation=max_rotation,
        rotation_probability=rotation_probability,
        enable_elastic=enable_elastic,
        elastic_alpha=elastic_alpha,
        elastic_sigma=max(0.1, elastic_sigma),
        elastic_probability=elastic_probability,
        enable_rect_erasing=enable_rect_erasing,
        rect_erase_height=max(1, rect_erase_height),
        rect_erase_width=max(1, rect_erase_width),
        rect_erasing_probability=rect_erasing_probability,
    )


def build_augmentation_from_search_state(
    reference: AugmentationConfig,
    state: AugmentationSearchState,
    search_config: AugmentationSearchConfig,
    *,
    image_height: int | None = None,
    image_width: int | None = None,
) -> AugmentationConfig:
    return build_parameter_scaled_augmentation(
        reference,
        state.parameter_values,
        search_config,
        image_height=image_height,
        image_width=image_width,
    )


def _format_probability(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _format_magnitude(value: float, *, unit: str = "", digits: int = 2) -> str:
    rounded = round(value)
    if abs(value - rounded) <= 1e-9:
        return f"{int(rounded)}{unit}"
    return f"{value:.{digits}f}{unit}"


def _format_scale_range(min_value: float, max_value: float) -> str:
    return f"{min_value:.2f}..{max_value:.2f}"


def _format_toggle(value: bool) -> str:
    return "on" if value else "off"


def format_augmentation_summary(augmentation: AugmentationConfig) -> str:
    return " | ".join(
        [
            (
                "translate["
                f"{_format_toggle(augmentation.enable_translation)}, "
                f"p={_format_probability(augmentation.translation_probability)}, "
                f"max={_format_magnitude(augmentation.max_translation, unit='px')}"
                "]"
            ),
            (
                "scale["
                f"{_format_toggle(augmentation.enable_scaling)}, "
                f"p={_format_probability(augmentation.scaling_probability)}, "
                f"x={_format_scale_range(augmentation.scale_min_x, augmentation.scale_max_x)}, "
                f"y={_format_scale_range(augmentation.scale_min_y, augmentation.scale_max_y)}"
                "]"
            ),
            (
                "rotate["
                f"{_format_toggle(augmentation.enable_rotation)}, "
                f"p={_format_probability(augmentation.rotation_probability)}, "
                f"max={_format_magnitude(augmentation.max_rotation, unit='deg', digits=1)}"
                "]"
            ),
            (
                "elastic["
                f"{_format_toggle(augmentation.enable_elastic)}, "
                f"p={_format_probability(augmentation.elastic_probability)}, "
                f"alpha={_format_magnitude(augmentation.elastic_alpha)}, "
                f"sigma={_format_magnitude(augmentation.elastic_sigma)}"
                "]"
            ),
            (
                "erase["
                f"{_format_toggle(augmentation.enable_rect_erasing)}, "
                f"p={_format_probability(augmentation.rect_erasing_probability)}, "
                f"h={augmentation.rect_erase_height}px, "
                f"w={augmentation.rect_erase_width}px"
                "]"
            ),
        ]
    )


def _float_changed(before: float, after: float) -> bool:
    return abs(before - after) > _SEARCH_TOLERANCE


def format_augmentation_changes(
    before: AugmentationConfig,
    after: AugmentationConfig,
) -> str:
    entries: list[str] = []

    translate_parts: list[str] = []
    if before.enable_translation != after.enable_translation:
        translate_parts.append(
            f"{_format_toggle(before.enable_translation)}->{_format_toggle(after.enable_translation)}"
        )
    if _float_changed(before.translation_probability, after.translation_probability):
        translate_parts.append(
            "p "
            f"{_format_probability(before.translation_probability)}->"
            f"{_format_probability(after.translation_probability)}"
        )
    if _float_changed(before.max_translation, after.max_translation):
        translate_parts.append(
            "max "
            f"{_format_magnitude(before.max_translation, unit='px')}->"
            f"{_format_magnitude(after.max_translation, unit='px')}"
        )
    if translate_parts:
        entries.append("translate: " + ", ".join(translate_parts))

    scale_parts: list[str] = []
    if before.enable_scaling != after.enable_scaling:
        scale_parts.append(
            f"{_format_toggle(before.enable_scaling)}->{_format_toggle(after.enable_scaling)}"
        )
    if _float_changed(before.scaling_probability, after.scaling_probability):
        scale_parts.append(
            "p "
            f"{_format_probability(before.scaling_probability)}->"
            f"{_format_probability(after.scaling_probability)}"
        )
    if _float_changed(before.scale_min_x, after.scale_min_x) or _float_changed(
        before.scale_max_x, after.scale_max_x
    ):
        scale_parts.append(
            "x "
            f"{_format_scale_range(before.scale_min_x, before.scale_max_x)}->"
            f"{_format_scale_range(after.scale_min_x, after.scale_max_x)}"
        )
    if _float_changed(before.scale_min_y, after.scale_min_y) or _float_changed(
        before.scale_max_y, after.scale_max_y
    ):
        scale_parts.append(
            "y "
            f"{_format_scale_range(before.scale_min_y, before.scale_max_y)}->"
            f"{_format_scale_range(after.scale_min_y, after.scale_max_y)}"
        )
    if scale_parts:
        entries.append("scale: " + ", ".join(scale_parts))

    rotate_parts: list[str] = []
    if before.enable_rotation != after.enable_rotation:
        rotate_parts.append(
            f"{_format_toggle(before.enable_rotation)}->{_format_toggle(after.enable_rotation)}"
        )
    if _float_changed(before.rotation_probability, after.rotation_probability):
        rotate_parts.append(
            "p "
            f"{_format_probability(before.rotation_probability)}->"
            f"{_format_probability(after.rotation_probability)}"
        )
    if _float_changed(before.max_rotation, after.max_rotation):
        rotate_parts.append(
            "max "
            f"{_format_magnitude(before.max_rotation, unit='deg', digits=1)}->"
            f"{_format_magnitude(after.max_rotation, unit='deg', digits=1)}"
        )
    if rotate_parts:
        entries.append("rotate: " + ", ".join(rotate_parts))

    elastic_parts: list[str] = []
    if before.enable_elastic != after.enable_elastic:
        elastic_parts.append(
            f"{_format_toggle(before.enable_elastic)}->{_format_toggle(after.enable_elastic)}"
        )
    if _float_changed(before.elastic_probability, after.elastic_probability):
        elastic_parts.append(
            "p "
            f"{_format_probability(before.elastic_probability)}->"
            f"{_format_probability(after.elastic_probability)}"
        )
    if _float_changed(before.elastic_alpha, after.elastic_alpha):
        elastic_parts.append(
            "alpha "
            f"{_format_magnitude(before.elastic_alpha)}->"
            f"{_format_magnitude(after.elastic_alpha)}"
        )
    if _float_changed(before.elastic_sigma, after.elastic_sigma):
        elastic_parts.append(
            "sigma "
            f"{_format_magnitude(before.elastic_sigma)}->"
            f"{_format_magnitude(after.elastic_sigma)}"
        )
    if elastic_parts:
        entries.append("elastic: " + ", ".join(elastic_parts))

    erase_parts: list[str] = []
    if before.enable_rect_erasing != after.enable_rect_erasing:
        erase_parts.append(
            f"{_format_toggle(before.enable_rect_erasing)}->{_format_toggle(after.enable_rect_erasing)}"
        )
    if _float_changed(before.rect_erasing_probability, after.rect_erasing_probability):
        erase_parts.append(
            "p "
            f"{_format_probability(before.rect_erasing_probability)}->"
            f"{_format_probability(after.rect_erasing_probability)}"
        )
    if before.rect_erase_height != after.rect_erase_height:
        erase_parts.append(f"h {before.rect_erase_height}px->{after.rect_erase_height}px")
    if before.rect_erase_width != after.rect_erase_width:
        erase_parts.append(f"w {before.rect_erase_width}px->{after.rect_erase_width}px")
    if erase_parts:
        entries.append("erase: " + ", ".join(erase_parts))

    return "; ".join(entries) if entries else "none"


@nnx.jit
def _logits_step(model: nnx.Module, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    model.eval()
    return model(batch["image"], deterministic=True)


def _prediction_margins(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    row_indices = jnp.arange(labels.shape[0])
    true_logits = logits[row_indices, labels]
    masked_logits = logits.at[row_indices, labels].set(-jnp.inf)
    best_other_logits = jnp.max(masked_logits, axis=1)
    return true_logits - best_other_logits


def _collect_anchor_set(
    model: nnx.Module,
    dataloader,
    *,
    max_examples: int,
    max_examples_per_class: int,
    num_classes: int,
    min_margin: float,
) -> AnchorSet | None:
    def _collect(require_margin: bool) -> AnchorSet | None:
        images = []
        labels = []
        margins = []
        class_counts = [0 for _ in range(num_classes)]
        num_examples = 0

        for batch in dataloader:
            collated = jax_collate(batch)
            logits = _logits_step(model, collated)
            preds = jnp.argmax(logits, axis=1)
            batch_margins = _prediction_margins(logits, collated["label"])
            anchor_mask = preds == collated["label"]
            if require_margin:
                anchor_mask = jnp.logical_and(anchor_mask, batch_margins >= min_margin)

            if bool(jnp.any(anchor_mask)):
                selected_images = np.asarray(jax.device_get(collated["image"][anchor_mask]))
                selected_labels = np.asarray(jax.device_get(collated["label"][anchor_mask]))
                selected_margins = np.asarray(jax.device_get(batch_margins[anchor_mask]))
                for class_id in range(num_classes):
                    remaining_total = max_examples - num_examples
                    remaining_class = max_examples_per_class - class_counts[class_id]
                    if remaining_total <= 0:
                        break
                    if remaining_class <= 0:
                        continue

                    class_indices = np.flatnonzero(selected_labels == class_id)
                    if class_indices.size == 0:
                        continue

                    take = min(class_indices.size, remaining_total, remaining_class)
                    chosen_indices = class_indices[:take]
                    images.append(jnp.asarray(selected_images[chosen_indices]))
                    labels.append(jnp.asarray(selected_labels[chosen_indices]))
                    margins.append(jnp.asarray(selected_margins[chosen_indices]))
                    class_counts[class_id] += int(take)
                    num_examples += int(take)

                if num_examples >= max_examples:
                    break

        if num_examples == 0:
            return None

        anchor_images = jnp.concatenate(images, axis=0)[:max_examples]
        anchor_labels = jnp.concatenate(labels, axis=0)[:max_examples]
        anchor_margins = jnp.concatenate(margins, axis=0)[:max_examples]
        effective_class_counts = [0 for _ in range(num_classes)]
        for label in np.asarray(jax.device_get(anchor_labels)):
            effective_class_counts[int(label)] += 1
        return AnchorSet(
            images=anchor_images,
            labels=anchor_labels,
            class_counts=effective_class_counts,
            mean_margin=float(jnp.mean(anchor_margins)),
        )

    anchor = _collect(require_margin=True)
    if anchor is not None or min_margin <= _SEARCH_TOLERANCE:
        return anchor
    return _collect(require_margin=False)


def _iter_anchor_batches(anchor: AnchorSet, batch_size: int):
    num_examples = int(anchor.labels.shape[0])
    for start in range(0, num_examples, batch_size):
        stop = min(num_examples, start + batch_size)
        yield {
            "image": anchor.images[start:stop],
            "label": anchor.labels[start:stop],
        }


def _candidate_eval_key(
    *,
    seed: int,
    epoch: int,
    parameter_index: int,
    normalized_value: float,
) -> jax.Array:
    key = jax.random.PRNGKey(seed)
    key = jax.random.fold_in(key, epoch)
    key = jax.random.fold_in(key, parameter_index)
    key = jax.random.fold_in(
        key,
        int(round(_clamp_strength(normalized_value) * 1_000_000)),
    )
    return key


def evaluate_invariance_candidate(
    model: nnx.Module,
    anchor: AnchorSet,
    augmentation: AugmentationConfig,
    *,
    batch_size: int,
    candidate_samples: int,
    required_retention: float | None = None,
    seed: int,
    epoch: int,
    parameter_index: int,
    normalized_value: float,
) -> CandidateEvaluation:
    search_augment_batch = getattr(augment_data_batch, "__wrapped__", augment_data_batch)
    augmentation_params = AugmentationParams(**augmentation.to_dict())
    base_key = _candidate_eval_key(
        seed=seed,
        epoch=epoch,
        parameter_index=parameter_index,
        normalized_value=normalized_value,
    )

    total_retained = 0
    total_examples = 0
    total_margin = 0.0
    total_possible_examples = int(candidate_samples * anchor.labels.shape[0])
    min_required_retained = None
    if required_retention is not None:
        min_required_retained = max(
            0.0,
            (required_retention - _SEARCH_TOLERANCE) * total_possible_examples,
        )

    for sample_index in range(candidate_samples):
        sample_key = jax.random.fold_in(base_key, sample_index)
        for batch_index, batch in enumerate(_iter_anchor_batches(anchor, batch_size)):
            batch_key = jax.random.fold_in(sample_key, batch_index)
            augmented_batch = search_augment_batch(batch, batch_key, augmentation_params)
            logits = _logits_step(model, augmented_batch)
            preds = jnp.argmax(logits, axis=1)
            margins = _prediction_margins(logits, batch["label"])
            total_retained += int(jnp.sum(preds == batch["label"]))
            total_examples += int(batch["label"].shape[0])
            total_margin += float(jnp.sum(margins))

            if min_required_retained is not None:
                remaining_examples = total_possible_examples - total_examples
                max_possible_retained = total_retained + remaining_examples
                if max_possible_retained < min_required_retained:
                    mean_margin = total_margin / total_examples if total_examples else 0.0
                    margin_retention = 1.0
                    if anchor.mean_margin > _SEARCH_TOLERANCE and total_examples > 0:
                        margin_retention = mean_margin / anchor.mean_margin
                    return CandidateEvaluation(
                        retention=max_possible_retained / total_possible_examples,
                        mean_margin=mean_margin,
                        margin_retention=margin_retention,
                    )

    if total_examples == 0:
        return CandidateEvaluation(
            retention=1.0,
            mean_margin=0.0,
            margin_retention=1.0,
        )

    mean_margin = total_margin / total_examples
    margin_retention = 1.0
    if anchor.mean_margin > _SEARCH_TOLERANCE:
        margin_retention = mean_margin / anchor.mean_margin

    return CandidateEvaluation(
        retention=total_retained / total_examples,
        mean_margin=mean_margin,
        margin_retention=margin_retention,
    )


def search_parameter_frontier(
    current_value: float,
    evaluate_candidate: Callable[[float], CandidateEvaluation],
    search_config: AugmentationSearchConfig,
) -> tuple[float, CandidateEvaluation]:
    min_step = search_config.min_strength_step
    coarse_step = search_config.strength_step
    eval_cache: dict[float, CandidateEvaluation] = {}

    def _evaluate(value: float) -> CandidateEvaluation:
        normalized_value = round(_clamp_strength(value), 8)
        if normalized_value not in eval_cache:
            eval_cache[normalized_value] = evaluate_candidate(normalized_value)
        return eval_cache[normalized_value]

    best_value = _clamp_strength(current_value)
    best_eval = _evaluate(best_value)

    if not _is_acceptable_candidate(best_eval, search_config) and best_value > _SEARCH_TOLERANCE:
        zero_eval = _evaluate(0.0)
        if not _is_acceptable_candidate(zero_eval, search_config):
            return 0.0, zero_eval

        low = 0.0
        high = best_value
        best_value = 0.0
        best_eval = zero_eval
        while high - low > min_step + _SEARCH_TOLERANCE:
            mid = 0.5 * (low + high)
            mid_eval = _evaluate(mid)
            if _is_acceptable_candidate(mid_eval, search_config):
                low = mid
                best_value = mid
                best_eval = mid_eval
            else:
                high = mid
    elif not _is_acceptable_candidate(best_eval, search_config):
        return 0.0, best_eval

    candidate_value = min(1.0, best_value + coarse_step)
    if candidate_value <= best_value + _SEARCH_TOLERANCE:
        return best_value, best_eval

    while True:
        candidate_eval = _evaluate(candidate_value)
        if _is_acceptable_candidate(candidate_eval, search_config):
            best_value = candidate_value
            best_eval = candidate_eval
            if best_value >= 1.0 - _SEARCH_TOLERANCE:
                return 1.0, best_eval

            next_candidate = min(1.0, candidate_value + coarse_step)
            if next_candidate <= candidate_value + _SEARCH_TOLERANCE:
                return best_value, best_eval
            candidate_value = next_candidate
            continue

        high = candidate_value
        while high - best_value > min_step + _SEARCH_TOLERANCE:
            mid = 0.5 * (best_value + high)
            mid_eval = _evaluate(mid)
            if _is_acceptable_candidate(mid_eval, search_config):
                best_value = mid
                best_eval = mid_eval
            else:
                high = mid
        return best_value, best_eval


def search_parameter_candidates(
    current_value: float,
    candidate_values: Sequence[float],
    evaluate_candidate: Callable[[float], CandidateEvaluation],
    search_config: AugmentationSearchConfig,
) -> tuple[float, CandidateEvaluation]:
    eval_cache: dict[float, CandidateEvaluation] = {}

    def _evaluate(value: float) -> CandidateEvaluation:
        normalized_value = round(_clamp_strength(value), 8)
        if normalized_value not in eval_cache:
            eval_cache[normalized_value] = evaluate_candidate(normalized_value)
        return eval_cache[normalized_value]

    normalized_current = round(_clamp_strength(current_value), 8)
    normalized_candidates = sorted(
        {
            round(_clamp_strength(value), 8)
            for value in (*candidate_values, normalized_current)
        }
    )
    scored_candidates = [(value, _evaluate(value)) for value in normalized_candidates]
    acceptable_candidates = [
        (value, candidate)
        for value, candidate in scored_candidates
        if _is_acceptable_candidate(candidate, search_config)
    ]

    if acceptable_candidates:

        def _acceptable_score(
            item: tuple[float, CandidateEvaluation],
        ) -> tuple[float, float, float, float]:
            value, candidate = item
            retention_slack = candidate.retention - search_config.invariance_threshold
            margin_slack = (
                candidate.margin_retention - search_config.margin_retention_threshold
            )
            return (
                retention_slack**2 + margin_slack**2,
                abs(value - normalized_current),
                -candidate.retention,
                -candidate.margin_retention,
            )

        return min(acceptable_candidates, key=_acceptable_score)

    def _fallback_score(
        item: tuple[float, CandidateEvaluation],
    ) -> tuple[float, float, float]:
        value, candidate = item
        return (
            candidate.retention,
            candidate.margin_retention,
            -abs(value - normalized_current),
        )

    return max(scored_candidates, key=_fallback_score)


def search_augmentation_parameters(
    model: nnx.Module,
    train_eval_dataloader,
    reference_augmentation: AugmentationConfig,
    state: AugmentationSearchState,
    config: MnistVitAugSearchConfig,
    epoch: int,
) -> tuple[AugmentationSearchState, AnchorSet | None, list[ParameterSearchUpdate]]:
    if state.search_frozen:
        return (
            replace(
                state,
                last_updated_parameters=[],
                last_event="search_frozen",
            ),
            None,
            [],
        )

    max_examples_per_class = max(
        1,
        min(
            config.search.anchor_examples_per_class,
            int(np.ceil(config.search.anchor_max_examples / config.model.num_classes)),
        ),
    )
    anchor = _collect_anchor_set(
        model,
        train_eval_dataloader,
        max_examples=config.search.anchor_max_examples,
        max_examples_per_class=max_examples_per_class,
        num_classes=config.model.num_classes,
        min_margin=config.search.anchor_min_margin,
    )
    if anchor is None:
        return (
            replace(
                state,
                last_anchor_size=0,
                last_anchor_num_classes=0,
                last_anchor_min_class_count=0,
                last_anchor_mean_margin=0.0,
                last_retention=1.0,
                last_margin_retention=1.0,
                last_updated_parameters=[],
                last_event="no_anchor_examples",
            ),
            None,
            [],
        )

    parameter_values = dict(state.parameter_values)
    updates: list[ParameterSearchUpdate] = []
    search_batch_size = max(1, config.training.batch_size)
    effective_search_config = _effective_search_config(config.search, state)

    for parameter_index, parameter_name in enumerate(_SEARCH_PARAMETER_ORDER):
        if not _parameter_is_searchable(parameter_name, parameter_values):
            continue

        def _evaluate_candidate(normalized_value: float) -> CandidateEvaluation:
            candidate_values = dict(parameter_values)
            candidate_values[parameter_name] = normalized_value
            candidate_augmentation = build_parameter_scaled_augmentation(
                reference_augmentation,
                candidate_values,
                config.search,
                image_height=config.model.height,
                image_width=config.model.width,
            )
            return evaluate_invariance_candidate(
                model,
                anchor,
                candidate_augmentation,
                batch_size=search_batch_size,
                candidate_samples=config.search.candidate_samples,
                required_retention=config.search.invariance_threshold,
                seed=config.seed,
                epoch=epoch,
                parameter_index=parameter_index,
                normalized_value=normalized_value,
            )

        old_value = parameter_values[parameter_name]
        if parameter_name in _NON_MONOTONIC_GRID_POINTS:
            num_points = _NON_MONOTONIC_GRID_POINTS[parameter_name]
            candidate_values = np.linspace(0.0, 1.0, num_points)
            new_value, candidate_eval = search_parameter_candidates(
                old_value,
                candidate_values,
                _evaluate_candidate,
                effective_search_config,
            )
        else:
            new_value, candidate_eval = search_parameter_frontier(
                old_value,
                _evaluate_candidate,
                effective_search_config,
            )
        parameter_values[parameter_name] = new_value
        if abs(new_value - old_value) > _SEARCH_TOLERANCE:
            updates.append(
                ParameterSearchUpdate(
                    parameter=parameter_name,
                    old_value=old_value,
                    new_value=new_value,
                    retention=candidate_eval.retention,
                    mean_margin=candidate_eval.mean_margin,
                    margin_retention=candidate_eval.margin_retention,
                )
            )

    final_augmentation = build_parameter_scaled_augmentation(
        reference_augmentation,
        parameter_values,
        config.search,
        image_height=config.model.height,
        image_width=config.model.width,
    )
    final_eval = evaluate_invariance_candidate(
        model,
        anchor,
        final_augmentation,
        batch_size=search_batch_size,
        candidate_samples=config.search.candidate_samples,
        required_retention=config.search.invariance_threshold,
        seed=config.seed,
        epoch=epoch,
        parameter_index=len(_SEARCH_PARAMETER_ORDER),
        normalized_value=0.0,
    )
    non_zero_class_counts = [count for count in anchor.class_counts if count > 0]
    next_state = replace(
        state,
        parameter_values=parameter_values,
        last_anchor_size=int(anchor.labels.shape[0]),
        last_anchor_num_classes=len(non_zero_class_counts),
        last_anchor_min_class_count=min(non_zero_class_counts) if non_zero_class_counts else 0,
        last_anchor_mean_margin=anchor.mean_margin,
        last_retention=final_eval.retention,
        last_margin_retention=final_eval.margin_retention,
        last_updated_parameters=[update.parameter for update in updates],
        last_event="search_updated" if updates else "search_stable",
    )
    return next_state, anchor, updates


def _to_numpy_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    def _convert(value: Any) -> Any:
        array = jax.device_get(value)
        try:
            array = jax.random.key_data(array)
        except TypeError:
            pass
        return np.asarray(array)

    return jax.tree_util.tree_map(_convert, tree)


def _model_payload(model: nnx.Module) -> Dict[str, Any]:
    keys_state, params_state = nnx.state(model, nnx.RngKey, ...)
    return {
        "keys": _to_numpy_tree(nnx.to_pure_dict(keys_state)),
        "state": _to_numpy_tree(nnx.to_pure_dict(params_state)),
    }


def _restore_model_payload(model: nnx.Module, payload: Dict[str, Any]) -> None:
    keys_state, params_state = nnx.state(model, nnx.RngKey, ...)
    restored_keys = jax.tree_util.tree_map(
        lambda x: jax.random.wrap_key_data(x)
        if isinstance(x, np.ndarray)
        and x.dtype == np.uint32
        and x.shape
        and x.shape[-1] == 2
        else x,
        payload["keys"],
    )
    nnx.replace_by_pure_dict(keys_state, restored_keys)
    nnx.replace_by_pure_dict(params_state, payload["state"])
    nnx.update(model, keys_state, params_state)


def create_training_snapshot(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    rng_key: jax.Array,
) -> bytes:
    payload = {
        "model": _model_payload(model),
        "optimizer": _to_numpy_tree(nnx.to_pure_dict(nnx.state(optimizer))),
        "rng_key": np.asarray(jax.random.key_data(rng_key)),
    }
    return serialization.to_bytes(payload)


def restore_training_snapshot(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    snapshot: bytes,
) -> jax.Array:
    template = {
        "model": _model_payload(model),
        "optimizer": _to_numpy_tree(nnx.to_pure_dict(nnx.state(optimizer))),
        "rng_key": np.asarray(jax.random.key_data(jax.random.PRNGKey(0))),
    }
    restored_payload = serialization.from_bytes(template, snapshot)
    _restore_model_payload(model, restored_payload["model"])
    optimizer_state = nnx.state(optimizer)
    nnx.replace_by_pure_dict(optimizer_state, restored_payload["optimizer"])
    nnx.update(optimizer, optimizer_state)
    return jax.random.wrap_key_data(restored_payload["rng_key"])


def _probe_baseline_path(checkpoint_dir: str) -> Path:
    return Path(checkpoint_dir) / "probe_baseline.msgpack"


def save_probe_baseline_model(model: nnx.Module, checkpoint_dir: str) -> Path:
    path = _probe_baseline_path(checkpoint_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(_model_payload(model)))
    return path


def load_probe_baseline_model(model: nnx.Module, checkpoint_dir: str) -> nnx.Module:
    path = _probe_baseline_path(checkpoint_dir)
    if not path.exists():
        raise FileNotFoundError(f"Probe baseline checkpoint not found at {path}")
    restored_payload = serialization.from_bytes(_model_payload(model), path.read_bytes())
    _restore_model_payload(model, restored_payload)
    return model


def clear_probe_baseline_model(checkpoint_dir: str) -> None:
    path = _probe_baseline_path(checkpoint_dir)
    path.unlink(missing_ok=True)


def _search_state_path(checkpoint_dir: str) -> Path:
    return Path(checkpoint_dir) / "augmentation_search_state.json"


def load_search_state(
    checkpoint_dir: str,
    search_config: AugmentationSearchConfig,
    output_dir: str | Path | None = None,
) -> AugmentationSearchState:
    del output_dir
    path = _search_state_path(checkpoint_dir)
    default_state = default_search_state(search_config)
    if not path.exists():
        return default_state

    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    raw_version = raw_payload.get("search_version")
    if raw_version == 3 and isinstance(raw_payload.get("parameter_values"), dict):
        payload = default_state.to_dict()
        payload.update(raw_payload)
        payload["search_version"] = _CURRENT_SEARCH_VERSION
        state = AugmentationSearchState(**payload)
        state.validate()
        return state

    if raw_version != _CURRENT_SEARCH_VERSION:
        return replace(default_state, last_event="reset_search_state")

    payload = default_state.to_dict()
    payload.update(raw_payload)
    state = AugmentationSearchState(**payload)
    state.validate()
    return state


def save_search_state(checkpoint_dir: str, state: AugmentationSearchState) -> Path:
    state.validate()
    path = _search_state_path(checkpoint_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _search_history_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "augmentation_search_history.jsonl"


def append_search_history_event(
    output_dir: str | Path,
    *,
    epoch: int,
    anchor_class_counts: Sequence[int],
    anchor_size: int,
    anchor_mean_margin: float,
    retention: float,
    margin_retention: float,
    updates: Sequence[ParameterSearchUpdate],
    augmentation_before: AugmentationConfig,
    augmentation_after: AugmentationConfig,
    augmentation_summary: str,
    augmentation_changes: str,
    state_before: AugmentationSearchState,
    state_after: AugmentationSearchState,
) -> Path:
    history_path = _search_history_path(output_dir)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "event": state_after.last_event,
        "anchor_class_counts": list(anchor_class_counts),
        "anchor_size": anchor_size,
        "anchor_mean_margin": anchor_mean_margin,
        "retention": retention,
        "margin_retention": margin_retention,
        "updates": [
            {
                "parameter": update.parameter,
                "old_value": update.old_value,
                "new_value": update.new_value,
                "retention": update.retention,
                "mean_margin": update.mean_margin,
                "margin_retention": update.margin_retention,
            }
            for update in updates
        ],
        "augmentation_before": augmentation_before.to_dict(),
        "augmentation_after": augmentation_after.to_dict(),
        "augmentation_summary": augmentation_summary,
        "augmentation_changes": augmentation_changes,
        "state_before": state_before.to_dict(),
        "state_after": state_after.to_dict(),
    }
    with history_path.open("a", encoding="utf-8") as fout:
        fout.write(_stable_json(payload) + "\n")
    return history_path


def _count_errors(accuracy: float, num_examples: int) -> int:
    return int(round((1.0 - accuracy) * num_examples))


def _evaluate_dataloader(
    model: nnx.Module,
    metrics: nnx.MultiMetric,
    dataloader,
) -> tuple[dict[str, float], int]:
    metrics.reset()
    num_batches = 0
    num_examples = 0
    for batch in dataloader:
        collated = jax_collate(batch)
        eval_step(model, metrics, collated)
        num_batches += 1
        num_examples += int(collated["label"].shape[0])
    if num_batches == 0:
        raise ValueError("evaluation dataloader did not yield any batches.")
    computed = {name: float(value.item()) for name, value in metrics.compute().items()}
    metrics.reset()
    return computed, num_examples


def save_augsearch_metrics(
    metrics_history: Dict[str, list],
    epoch: int,
    *,
    output_csv: str,
) -> None:
    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv_path.is_file()

    with output_csv_path.open(mode="a", newline="", encoding="utf-8") as csv_file:
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
                    "train_error_count",
                    "search_anchor_size",
                    "search_anchor_num_classes",
                    "search_anchor_min_class_count",
                    "search_anchor_mean_margin",
                    "search_retention",
                    "search_margin_retention",
                    "search_num_updates",
                    "search_updated_parameters",
                    "search_augmentation_changes",
                    "search_augmentation",
                    "search_event",
                ]
            )
        writer.writerow(
            [
                epoch,
                metrics_history["train_accuracy"][-1],
                metrics_history["train_accuracy_mean"][-1],
                metrics_history["train_accuracy_spread"][-1],
                metrics_history["test_accuracy"][-1],
                metrics_history["test_accuracy_mean"][-1],
                metrics_history["test_accuracy_spread"][-1],
                metrics_history["train_error_count"][-1],
                metrics_history["search_anchor_size"][-1],
                metrics_history["search_anchor_num_classes"][-1],
                metrics_history["search_anchor_min_class_count"][-1],
                metrics_history["search_anchor_mean_margin"][-1],
                metrics_history["search_retention"][-1],
                metrics_history["search_margin_retention"][-1],
                metrics_history["search_num_updates"][-1],
                metrics_history["search_updated_parameters"][-1],
                metrics_history["search_augmentation_changes"][-1],
                metrics_history["search_augmentation"][-1],
                metrics_history["search_event"][-1],
            ]
        )


def _apply_search_overrides(
    config: MnistVitAugSearchConfig, args: argparse.Namespace
) -> MnistVitAugSearchConfig:
    if args.anchor_max_examples is not None:
        config.search.anchor_max_examples = args.anchor_max_examples
    if args.anchor_examples_per_class is not None:
        config.search.anchor_examples_per_class = args.anchor_examples_per_class
    if args.anchor_min_margin is not None:
        config.search.anchor_min_margin = args.anchor_min_margin
    if args.invariance_threshold is not None:
        config.search.invariance_threshold = args.invariance_threshold
    if args.margin_retention_threshold is not None:
        config.search.margin_retention_threshold = args.margin_retention_threshold
    if args.candidate_samples is not None:
        config.search.candidate_samples = args.candidate_samples
    if args.strength_step is not None:
        config.search.strength_step = args.strength_step
    if args.min_strength_step is not None:
        config.search.min_strength_step = args.min_strength_step
    if args.search_every_n_epochs is not None:
        config.search.search_every_n_epochs = args.search_every_n_epochs
    if args.max_strength is not None:
        config.search.max_strength = args.max_strength
    config.validate()
    return config


def parse_augsearch_args(
    args: Sequence[str] | None,
    *,
    description: str,
    default_onnx_output: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--onnx-output", type=str, default=default_onnx_output)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--anchor-max-examples", type=int, default=None)
    parser.add_argument("--anchor-examples-per-class", type=int, default=None)
    parser.add_argument("--anchor-min-margin", type=float, default=None)
    parser.add_argument("--invariance-threshold", type=float, default=None)
    parser.add_argument("--margin-retention-threshold", type=float, default=None)
    parser.add_argument("--candidate-samples", type=int, default=None)
    parser.add_argument("--strength-step", type=float, default=None)
    parser.add_argument("--min-strength-step", type=float, default=None)
    parser.add_argument("--search-every-n-epochs", type=int, default=None)
    parser.add_argument("--max-strength", type=float, default=None)
    parsed_args = parser.parse_args(args)
    parsed_args.default_onnx_output = default_onnx_output
    return parsed_args


def get_default_config() -> MnistVitAugSearchConfig:
    base_config = mnist_vit_lib.get_default_config()
    default_output_dir = os.path.abspath("./output/mnist_vit_augsearch/")
    default_onnx_dir = os.path.abspath("./onnx/")
    base_config.training.checkpoint_dir = os.path.abspath(
        "./data/mnist_vit_augsearch_checkpoints/"
    )
    base_config.training.output_dir = default_output_dir
    base_config.onnx.model_name = "mnist_vit_augsearch_model"
    base_config.onnx.output_path = os.path.join(
        default_onnx_dir, "mnist_vit_augsearch_model.onnx"
    )
    return MnistVitAugSearchConfig(
        seed=base_config.seed,
        training=base_config.training,
        model=base_config.model,
        onnx=base_config.onnx,
        search=AugmentationSearchConfig(),
    )


def create_model(model_config: MnistVitModelConfig, seed: int) -> nnx.Module:
    return mnist_vit_lib.create_model(model_config, seed)


def _append_augsearch_benchmark_record(
    config: MnistVitAugSearchConfig,
    metrics_history: dict[str, list[float]],
    final_search_state: AugmentationSearchState,
) -> Path:
    record = build_benchmark_record(config, metrics_history)
    record["search_config"] = config.search.to_dict()
    record["final_search_state"] = final_search_state.to_dict()
    record["augmentation_strategy"] = "frozen_model_invariance_search"

    output_path = config.benchmark_memory_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fout:
        fout.write(_stable_json(record) + "\n")
    return output_path


def _initialize_metrics_history() -> Dict[str, list]:
    return {
        "train_loss": [],
        "train_accuracy": [],
        "train_accuracy_mean": [],
        "train_accuracy_spread": [],
        "train_online_loss": [],
        "train_online_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "test_accuracy_mean": [],
        "test_accuracy_spread": [],
        "train_error_count": [],
        "search_anchor_size": [],
        "search_anchor_num_classes": [],
        "search_anchor_min_class_count": [],
        "search_anchor_mean_margin": [],
        "search_retention": [],
        "search_margin_retention": [],
        "search_num_updates": [],
        "search_updated_parameters": [],
        "search_augmentation_changes": [],
        "search_augmentation": [],
        "search_event": [],
    }


def _format_search_updates(updates: Sequence[ParameterSearchUpdate]) -> str:
    if not updates:
        return "none"
    return "; ".join(
        f"{update.parameter} {update.old_value:.3f}->{update.new_value:.3f}"
        for update in updates
    )


def train_model(
    model: nnx.Module,
    start_epoch: int,
    metrics: nnx.MultiMetric,
    config: MnistVitAugSearchConfig,
    train_dataloader,
    test_dataloader,
    rng_key: jax.Array,
) -> Dict[str, list]:
    output_dir = str(config.artifact_dir())
    metrics_csv_path = str(config.metrics_csv_path())
    metrics_fig_path = str(config.metrics_plot_path())
    reference_augmentation = config.training.augmentation
    train_eval_dataloader = mnist_training_lib._build_clean_eval_dataloader(train_dataloader)

    metrics_history = _initialize_metrics_history()

    initial_lr = config.training.base_learning_rate
    initial_weight_decay = config.training.weight_decay
    optimizer = create_optimizer(model, initial_lr, initial_weight_decay)
    search_state = load_search_state(
        config.training.checkpoint_dir,
        config.search,
        output_dir=config.artifact_dir(),
    )
    if search_state.last_event == "reset_search_state":
        print(
            "[search] found an incompatible legacy search state. "
            "Restarting the augmentation search with the frozen-model invariance strategy."
        )

    final_epoch = start_epoch + config.training.num_epochs_to_train_now - 1
    for epoch in range(start_epoch, start_epoch + config.training.num_epochs_to_train_now):
        learning_rate = lr_schedule(epoch, config.training)
        weight_decay = config.training.weight_decay
        state_before = search_state
        effective_search_config = _effective_search_config(config.search, state_before)
        refresh_artifacts = _should_refresh_artifacts(epoch, final_epoch=final_epoch)
        augmentation_before = build_augmentation_from_search_state(
            reference_augmentation,
            state_before,
            config.search,
            image_height=config.model.height,
            image_width=config.model.width,
        )
        if _should_run_search_epoch(epoch, state_before, config.search):
            search_state, anchor_set, search_updates = search_augmentation_parameters(
                model,
                train_eval_dataloader,
                reference_augmentation,
                search_state,
                config,
                epoch,
            )
        else:
            search_state = replace(
                search_state,
                last_updated_parameters=[],
                last_event="search_skipped_interval",
            )
            anchor_set = None
            search_updates = []
        augmentation_after = build_augmentation_from_search_state(
            reference_augmentation,
            search_state,
            config.search,
            image_height=config.model.height,
            image_width=config.model.width,
        )
        updated_parameters = _format_search_updates(search_updates)
        augmentation_changes = format_augmentation_changes(
            augmentation_before,
            augmentation_after,
        )
        augmentation_summary = format_augmentation_summary(augmentation_after)
        if search_state.last_event == "search_skipped_interval":
            search_status = "skipped; reusing current augmentation"
        elif search_state.last_event == "search_frozen":
            search_status = "search frozen; reusing current augmentation"
        else:
            search_status = (
                f"anchor size: {search_state.last_anchor_size}, "
                f"classes: {search_state.last_anchor_num_classes}, "
                f"min/class: {search_state.last_anchor_min_class_count}, "
                f"anchor margin: {search_state.last_anchor_mean_margin:.4f}, "
                f"retention: {search_state.last_retention:.4f}, "
                f"margin retention: {search_state.last_margin_retention:.4f}"
            )
        print(
            f"[search] epoch: {epoch}, {search_status}"
        )
        print(
            f"[search] control: step {effective_search_config.strength_step:.3f}, "
            f"min step {effective_search_config.min_strength_step:.3f}, "
            f"stale train epochs {state_before.epochs_since_train_improvement}, "
            f"frozen {'yes' if state_before.search_frozen else 'no'}, "
            f"interval {config.search.search_every_n_epochs}"
        )
        print(f"[search] changes: {augmentation_changes}")
        print(f"[search] augmentation: {augmentation_summary}")

        print(
            f"Epoch: {epoch}, learning rate: {learning_rate:.6e}, "
            f"search event: {search_state.last_event}"
        )

        epoch_augmentation = build_augmentation_from_search_state(
            reference_augmentation,
            search_state,
            config.search,
            image_height=config.model.height,
            image_width=config.model.width,
        )
        augmentation_params = AugmentationParams(**epoch_augmentation.to_dict())

        metrics.reset()
        last_train_batch = None
        for batch in train_dataloader:
            batch = jax_collate(batch)
            rng_key, dropout_rng = jax.random.split(rng_key)
            batch = augment_data_batch(batch, dropout_rng, augmentation_params)
            train_step(model, optimizer, metrics, batch, learning_rate, weight_decay)
            last_train_batch = batch

        if last_train_batch is None:
            raise ValueError(
                "train_dataloader did not yield any batches. "
                "Reduce batch_size or disable drop_last."
            )

        if refresh_artifacts:
            visualize_augmented_images(
                last_train_batch,
                epoch,
                num_images=9,
                output_dir=output_dir,
            )

        train_online_metrics = metrics.compute()
        metrics_history["train_online_loss"].append(train_online_metrics["loss"].item())
        metrics_history["train_online_accuracy"].append(
            train_online_metrics["accuracy"].item()
        )
        metrics_history["train_loss"].append(train_online_metrics["loss"].item())
        print(
            f"[train-online] epoch: {epoch}, "
            f"loss: {metrics_history['train_online_loss'][-1]:.4f}, "
            f"accuracy: {metrics_history['train_online_accuracy'][-1]:.4f}"
        )

        train_eval_metrics, num_train_examples = _evaluate_dataloader(
            model, metrics, train_eval_dataloader
        )
        metrics_history["train_accuracy"].append(train_eval_metrics["accuracy"])
        train_error_count = _count_errors(
            train_eval_metrics["accuracy"], num_train_examples
        )
        metrics_history["train_error_count"].append(train_error_count)
        print(
            f"[train] epoch: {epoch}, clean accuracy: "
            f"{train_eval_metrics['accuracy']:.4f}, errors: {train_error_count}"
        )
        search_state, train_plateau_metric, freeze_message = update_search_train_plateau_state(
            search_state,
            metrics_history["train_accuracy"],
            epoch,
            config.search,
        )
        if train_plateau_metric is not None:
            print(
                f"[search] train control: metric {train_plateau_metric:.4f}, "
                f"best {search_state.best_train_accuracy_metric:.4f}, "
                f"stale epochs {search_state.epochs_since_train_improvement}, "
                f"frozen {'yes' if search_state.search_frozen else 'no'}"
            )
        if freeze_message is not None:
            print(freeze_message)

        test_metrics, _ = _evaluate_dataloader(model, metrics, test_dataloader)
        metrics_history["test_loss"].append(test_metrics["loss"])
        metrics_history["test_accuracy"].append(test_metrics["accuracy"])
        print(
            f"[test] epoch: {epoch}, loss: {test_metrics['loss']:.4f}, "
            f"accuracy: {test_metrics['accuracy']:.4f}"
        )

        for split in ("train", "test"):
            acc_key = f"{split}_accuracy"
            mean_key = f"{split}_accuracy_mean"
            spread_key = f"{split}_accuracy_spread"
            if len(metrics_history[acc_key]) >= 10:
                recent_accuracies = metrics_history[acc_key][-10:]
                mean_accuracy, spread_accuracy = compute_mean_and_spread(recent_accuracies)
                metrics_history[mean_key].append(mean_accuracy)
                metrics_history[spread_key].append(spread_accuracy)
            else:
                metrics_history[mean_key].append(0.0)
                metrics_history[spread_key].append(0.0)

        metrics_history["search_anchor_size"].append(search_state.last_anchor_size)
        metrics_history["search_anchor_num_classes"].append(
            search_state.last_anchor_num_classes
        )
        metrics_history["search_anchor_min_class_count"].append(
            search_state.last_anchor_min_class_count
        )
        metrics_history["search_anchor_mean_margin"].append(
            search_state.last_anchor_mean_margin
        )
        metrics_history["search_retention"].append(search_state.last_retention)
        metrics_history["search_margin_retention"].append(
            search_state.last_margin_retention
        )
        metrics_history["search_num_updates"].append(len(search_updates))
        metrics_history["search_updated_parameters"].append(updated_parameters)
        metrics_history["search_augmentation_changes"].append(augmentation_changes)
        metrics_history["search_augmentation"].append(augmentation_summary)
        metrics_history["search_event"].append(search_state.last_event)

        append_search_history_event(
            output_dir,
            epoch=epoch,
            anchor_class_counts=anchor_set.class_counts if anchor_set is not None else [],
            anchor_size=search_state.last_anchor_size,
            anchor_mean_margin=search_state.last_anchor_mean_margin,
            retention=search_state.last_retention,
            margin_retention=search_state.last_margin_retention,
            updates=search_updates,
            augmentation_before=augmentation_before,
            augmentation_after=augmentation_after,
            augmentation_summary=augmentation_summary,
            augmentation_changes=augmentation_changes,
            state_before=state_before,
            state_after=search_state,
        )

        if refresh_artifacts:
            visualize_incorrect_classifications(
                model, test_dataloader, epoch, output_dir=output_dir
            )

        save_augsearch_metrics(metrics_history, epoch, output_csv=metrics_csv_path)
        save_model(model, config.training.checkpoint_dir, epoch)
        save_search_state(config.training.checkpoint_dir, search_state)
        if refresh_artifacts:
            load_and_plot_test_accuracy_metrics(metrics_csv_path, metrics_fig_path)
        else:
            print(
                f"[artifacts] epoch: {epoch}, deferred visual refresh "
                f"(every {_ARTIFACT_REFRESH_INTERVAL} epochs)."
            )

    return metrics_history


def main(args: Sequence[str] | None = None) -> None:
    cli_args = parse_augsearch_args(
        args,
        description="Train and export the MNIST ViT augmentation-search example",
        default_onnx_output="onnx/mnist_vit_augsearch_model.onnx",
    )
    config = apply_common_overrides(get_default_config(), cli_args)
    config = _apply_search_overrides(config, cli_args)

    train_dataloader, test_dataloader = get_dataset_torch_dataloaders(
        config.training.batch_size,
        config.training.data_dir,
    )
    rng_key = jax.random.PRNGKey(config.seed)

    latest_checkpoint_epoch, start_epoch = resolve_checkpoint_resume(
        config.training.checkpoint_dir
    )
    config.training.start_epoch = start_epoch
    if latest_checkpoint_epoch is None:
        print("No checkpoint found. Starting from scratch.")
    else:
        print(
            f"Loaded checkpoint from epoch {latest_checkpoint_epoch}. "
            f"Continuing at epoch {start_epoch}."
        )

    config_path = config.write_json(config.config_output_path())
    print(f"Resolved config saved to {config_path}")
    print(f"Run artifacts will be written to {config.artifact_dir()}")

    model = create_model(config.model, config.seed)
    if latest_checkpoint_epoch is not None:
        try:
            model = load_model(
                model,
                config.training.checkpoint_dir,
                latest_checkpoint_epoch,
                config.seed,
            )
            print(f"Loaded model from epoch {latest_checkpoint_epoch}")
        except FileNotFoundError:
            print(
                f"Checkpoint for epoch {latest_checkpoint_epoch} not found, "
                "starting from scratch."
            )
            start_epoch = 0
            config.training.start_epoch = 0
            model = create_model(config.model, config.seed)

    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    if config.training.enable_training:
        metrics_history = train_model(
            model,
            start_epoch,
            metrics,
            config,
            train_dataloader,
            test_dataloader,
            rng_key,
        )
        final_search_state = load_search_state(
            config.training.checkpoint_dir,
            config.search,
            output_dir=config.artifact_dir(),
        )
        if metrics_history["test_accuracy"]:
            visualize_results(
                metrics_history,
                model,
                test_dataloader,
                start_epoch + len(metrics_history["test_accuracy"]) - 1,
                output_dir=str(config.artifact_dir()),
            )
        else:
            print("Training skipped because no epochs were scheduled.")
        benchmark_memory_path = _append_augsearch_benchmark_record(
            config, metrics_history, final_search_state
        )
        print(f"Benchmark memory updated at {benchmark_memory_path}")

    output_path = Path(config.onnx.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("Exporting model to ONNX...")
    onnx_model = to_onnx(model, config.onnx.input_shapes, config.onnx.input_params)
    onnx.save_model(onnx_model, str(output_path))
    print(f"Model exported to {output_path}")

    xs = [jax.random.normal(rng_key, (4, 28, 28, 1))]
    model.eval()
    result = allclose(model, str(output_path), xs, config.onnx.input_params)
    print(f"ONNX allclose result: {result}")
    test_onnx_model(str(output_path), test_dataloader)


if __name__ == "__main__":
    main()
