import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jaxamples.mnist_config import MnistExampleConfig


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _accuracy_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"best": None, "best_epoch": None, "final": None}

    best_epoch = max(range(len(values)), key=values.__getitem__)
    return {
        "best": float(values[best_epoch]),
        "best_epoch": int(best_epoch),
        "final": float(values[-1]),
    }


def build_benchmark_record(
    config: MnistExampleConfig, metrics_history: dict[str, list[float]]
) -> dict[str, Any]:
    augmentation = config.training.augmentation.to_dict()
    augmentation_fingerprint = hashlib.sha256(
        _stable_json(augmentation).encode("utf-8")
    ).hexdigest()[:16]

    train_summary = _accuracy_summary(metrics_history.get("train_accuracy", []))
    test_summary = _accuracy_summary(metrics_history.get("test_accuracy", []))

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": config.onnx.model_name,
        "seed": config.seed,
        "artifact_dir": str(config.artifact_dir()),
        "metrics_csv_path": str(config.metrics_csv_path()),
        "config_output_path": str(config.config_output_path()),
        "onnx_output_path": str(config.onnx.output_path),
        "num_trained_epochs": len(metrics_history.get("test_accuracy", [])),
        "train_accuracy_semantics": "clean_eval_deterministic",
        "augmentation_fingerprint": augmentation_fingerprint,
        "augmentation": augmentation,
        "model_config": (
            config.model.to_dict() if hasattr(config.model, "to_dict") else config.model
        ),
        "training_config": {
            "batch_size": config.training.batch_size,
            "base_learning_rate": config.training.base_learning_rate,
            "warmup_epochs": config.training.warmup_epochs,
            "weight_decay": config.training.weight_decay,
            "start_epoch": config.training.start_epoch,
            "num_epochs_to_train_now": config.training.num_epochs_to_train_now,
            "checkpoint_dir": config.training.checkpoint_dir,
            "output_dir": config.training.output_dir,
        },
        "best_train_accuracy": train_summary["best"],
        "best_train_epoch": train_summary["best_epoch"],
        "final_train_accuracy": train_summary["final"],
        "best_test_accuracy": test_summary["best"],
        "best_test_epoch": test_summary["best_epoch"],
        "final_test_accuracy": test_summary["final"],
    }


def append_benchmark_record(
    config: MnistExampleConfig,
    metrics_history: dict[str, list[float]],
    *,
    output_path: str | Path | None = None,
) -> Path:
    memory_path = Path(output_path) if output_path is not None else config.benchmark_memory_path()
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    record = build_benchmark_record(config, metrics_history)
    with memory_path.open("a", encoding="utf-8") as fout:
        fout.write(_stable_json(record) + "\n")
    return memory_path


def load_benchmark_records(path: str | Path) -> list[dict[str, Any]]:
    memory_path = Path(path)
    if not memory_path.exists():
        return []

    records = []
    with memory_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records
