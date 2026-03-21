#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.image import scale_and_translate

from jaxamples import mnist_training, mnist_vit


@nnx.jit
def logits_step(model: nnx.Module, images: jnp.ndarray) -> jnp.ndarray:
    model.eval()
    return model(images, deterministic=True)


def _transform_translate(image: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    return scale_and_translate(
        image=image,
        shape=image.shape,
        spatial_dims=(0, 1),
        scale=jnp.ones((2,), dtype=image.dtype),
        translation=jnp.array([dy, dx], dtype=image.dtype),
        method="linear",
        antialias=True,
    )


def _transform_scale(
    image: jnp.ndarray, scale_x: float, scale_y: float, dx: float = 0.0, dy: float = 0.0
) -> jnp.ndarray:
    return scale_and_translate(
        image=image,
        shape=image.shape,
        spatial_dims=(0, 1),
        scale=jnp.array([scale_y, scale_x], dtype=image.dtype),
        translation=jnp.array([dy, dx], dtype=image.dtype),
        method="linear",
        antialias=True,
    )


def _transform_rotate(image: jnp.ndarray, angle_deg: float) -> jnp.ndarray:
    return mnist_training.rotate_image(image, angle_deg)


def _transform_combined(
    image: jnp.ndarray,
    dx: float,
    dy: float,
    angle_deg: float,
    scale: float,
) -> jnp.ndarray:
    corrected = _transform_scale(image, scale_x=scale, scale_y=scale, dx=dx, dy=dy)
    return _transform_rotate(corrected, angle_deg)


def _collect_misclassifications(
    model: nnx.Module,
    test_dataloader,
) -> list[dict[str, Any]]:
    misclassified: list[dict[str, Any]] = []
    sample_index = 0
    for batch in test_dataloader:
        batch = mnist_vit.jax_collate(batch)
        logits = logits_step(model, batch["image"])
        probs = jax.nn.softmax(logits, axis=1)
        preds = jnp.argmax(logits, axis=1)
        labels = batch["label"]
        mask = preds != labels
        mask_np = np.asarray(mask)
        for local_idx in np.nonzero(mask_np)[0]:
            label = int(labels[local_idx])
            pred = int(preds[local_idx])
            misclassified.append(
                {
                    "sample_index": sample_index + int(local_idx),
                    "image": batch["image"][local_idx],
                    "label": label,
                    "prediction": pred,
                    "true_probability": float(probs[local_idx, label]),
                    "pred_probability": float(probs[local_idx, pred]),
                }
            )
        sample_index += int(batch["image"].shape[0])
    return misclassified


def _evaluate_candidates(
    model: nnx.Module,
    image: jnp.ndarray,
    label: int,
    family: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not candidates:
        return None

    transformed_images = jnp.stack([candidate["image"] for candidate in candidates], axis=0)
    logits = logits_step(model, transformed_images)
    probs = jax.nn.softmax(logits, axis=1)
    preds = jnp.argmax(logits, axis=1)

    success_indices = np.nonzero(np.asarray(preds == label))[0]
    if len(success_indices) == 0:
        return None

    best_idx = max(
        success_indices,
        key=lambda idx: float(probs[idx, label]),
    )
    best_candidate = candidates[int(best_idx)]
    best_params = dict(best_candidate["params"])
    best_params["family"] = family
    return {
        "family": family,
        "params": best_params,
        "true_probability": float(probs[best_idx, label]),
        "predicted_label": int(preds[best_idx]),
    }


def _round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _build_family_candidates(image: jnp.ndarray) -> dict[str, list[dict[str, Any]]]:
    translations = []
    for dx, dy in product(range(-4, 5), repeat=2):
        if dx == 0 and dy == 0:
            continue
        translations.append(
            {
                "params": {"dx": dx, "dy": dy},
                "image": _transform_translate(image, dx=float(dx), dy=float(dy)),
            }
        )

    rotations = []
    for angle in range(-14, 15, 2):
        if angle == 0:
            continue
        rotations.append(
            {
                "params": {"angle_deg": angle},
                "image": _transform_rotate(image, angle_deg=float(angle)),
            }
        )

    isotropic_scales = []
    for scale in [0.85, 0.9, 0.95, 1.05, 1.1, 1.15]:
        isotropic_scales.append(
            {
                "params": {"scale": scale},
                "image": _transform_scale(image, scale_x=scale, scale_y=scale),
            }
        )

    anisotropic_scales = []
    scale_values = [0.9, 0.95, 1.0, 1.05, 1.1]
    for scale_x, scale_y in product(scale_values, scale_values):
        if scale_x == 1.0 and scale_y == 1.0:
            continue
        anisotropic_scales.append(
            {
                "params": {"scale_x": scale_x, "scale_y": scale_y},
                "image": _transform_scale(image, scale_x=scale_x, scale_y=scale_y),
            }
        )

    combined_rigid = []
    for dx, dy, angle, scale in product(
        [-2.0, 0.0, 2.0],
        [-2.0, 0.0, 2.0],
        [-8.0, 0.0, 8.0],
        [0.95, 1.0, 1.05],
    ):
        if dx == 0.0 and dy == 0.0 and angle == 0.0 and scale == 1.0:
            continue
        combined_rigid.append(
            {
                "params": {
                    "dx": dx,
                    "dy": dy,
                    "angle_deg": angle,
                    "scale": scale,
                },
                "image": _transform_combined(
                    image, dx=dx, dy=dy, angle_deg=angle, scale=scale
                ),
            }
        )

    return {
        "translation": translations,
        "rotation": rotations,
        "isotropic_scale": isotropic_scales,
        "anisotropic_scale": anisotropic_scales,
        "combined_rigid": combined_rigid,
    }


def _aggregate_family_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "correction_rate": 0.0,
        }

    summary: dict[str, Any] = {
        "count": len(records),
    }
    for key in sorted(records[0]["params"].keys()):
        if key == "family":
            continue
        values = [record["params"][key] for record in records]
        if all(isinstance(value, (int, np.integer)) for value in values):
            counts = Counter(int(value) for value in values)
            summary[f"{key}_counts"] = dict(sorted(counts.items()))
            abs_counts = Counter(abs(int(value)) for value in values)
            summary[f"{key}_abs_counts"] = dict(sorted(abs_counts.items()))
        else:
            rounded_values = [_round_float(value) for value in values]
            counts = Counter(rounded_values)
            summary[f"{key}_counts"] = dict(sorted(counts.items()))
            if key.startswith("scale"):
                summary[f"{key}_min"] = min(rounded_values)
                summary[f"{key}_max"] = max(rounded_values)
            else:
                abs_counts = Counter(_round_float(abs(value)) for value in values)
                summary[f"{key}_abs_counts"] = dict(sorted(abs_counts.items()))
    true_probabilities = [record["true_probability"] for record in records]
    summary["true_probability_mean"] = _round_float(np.mean(true_probabilities), 6)
    summary["true_probability_min"] = _round_float(min(true_probabilities), 6)
    summary["true_probability_max"] = _round_float(max(true_probabilities), 6)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze which inverse-style image transforms rescue current MNIST ViT "
            "misclassifications, to derive candidate augmentation ranges."
        )
    )
    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        default=None,
        help="Checkpoint epoch to analyze. Defaults to the latest available checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/inverse_augmentation_analysis",
        help="Directory for JSON and CSV analysis artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = mnist_vit.get_default_config()
    latest_epoch = mnist_training.get_latest_checkpoint_epoch(config.training.checkpoint_dir)
    if latest_epoch is None:
        raise FileNotFoundError(
            f"No checkpoints found in {config.training.checkpoint_dir}."
        )
    checkpoint_epoch = latest_epoch if args.checkpoint_epoch is None else args.checkpoint_epoch

    model = mnist_vit.create_model(config.model, seed=config.seed)
    mnist_vit.load_model(
        model,
        config.training.checkpoint_dir,
        epoch=checkpoint_epoch,
        seed=config.seed,
    )

    _, test_dataloader = mnist_vit.get_dataset_torch_dataloaders(
        batch_size=config.training.batch_size,
        data_dir=config.training.data_dir,
    )
    misclassified = _collect_misclassifications(model, test_dataloader)

    detailed_rows: list[dict[str, Any]] = []
    family_successes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    best_combined_fixes: list[dict[str, Any]] = []

    for item in misclassified:
        family_candidates = _build_family_candidates(item["image"])
        for family, candidates in family_candidates.items():
            best_fix = _evaluate_candidates(
                model=model,
                image=item["image"],
                label=item["label"],
                family=family,
                candidates=candidates,
            )
            if best_fix is None:
                continue
            row = {
                "sample_index": item["sample_index"],
                "label": item["label"],
                "original_prediction": item["prediction"],
                "original_true_probability": _round_float(item["true_probability"], 6),
                "original_pred_probability": _round_float(item["pred_probability"], 6),
                "family": family,
                "corrected_true_probability": _round_float(
                    best_fix["true_probability"], 6
                ),
                **best_fix["params"],
            }
            detailed_rows.append(row)
            family_successes[family].append(best_fix)
            if family == "combined_rigid":
                best_combined_fixes.append(row)

    incorrect_count = len(misclassified)
    summary = {
        "checkpoint_epoch": checkpoint_epoch,
        "incorrect_count": incorrect_count,
        "families": {},
    }

    for family, successes in sorted(family_successes.items()):
        family_summary = _aggregate_family_summary(successes)
        family_summary["correction_rate"] = _round_float(
            family_summary["count"] / incorrect_count if incorrect_count else 0.0,
            6,
        )
        summary["families"][family] = family_summary

    summary_path = output_dir / f"mnist_vit_inverse_augmentation_summary_epoch{checkpoint_epoch}.json"
    details_path = output_dir / f"mnist_vit_inverse_augmentation_details_epoch{checkpoint_epoch}.csv"

    with summary_path.open("w", encoding="utf-8") as fout:
        json.dump(_to_jsonable(summary), fout, indent=2, sort_keys=True)
        fout.write("\n")

    fieldnames = sorted({key for row in detailed_rows for key in row.keys()})
    with details_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in detailed_rows:
            writer.writerow(_to_jsonable(row))

    print(json.dumps(_to_jsonable(summary), indent=2, sort_keys=True))
    print(f"\nWrote summary to {summary_path}")
    print(f"Wrote details to {details_path}")


if __name__ == "__main__":
    main()
