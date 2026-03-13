from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import onnx
from flax import nnx
from torch.utils.data import DataLoader

from jaxamples.mnist_config import MnistExampleConfig


ModelFactory = Callable[[Any, int], nnx.Module]
DataLoaderFactory = Callable[[int, str], Tuple[DataLoader, DataLoader]]
CheckpointResolver = Callable[[str], Tuple[Optional[int], int]]
CheckpointLoader = Callable[[nnx.Module, str, int, int], nnx.Module]
TrainFn = Callable[
    [nnx.Module, int, nnx.MultiMetric, Any, DataLoader, DataLoader, jax.Array],
    Dict[str, list],
]
VisualizeFn = Callable[[Dict[str, list], nnx.Module, DataLoader, int, str], None]
OnnxEvalFn = Callable[[str, DataLoader], None]


def _build_concrete_shapes(input_shapes: list[tuple[Any, ...]]) -> list[tuple[int, ...]]:
    concrete_shapes = []
    for shape in input_shapes:
        concrete_shapes.append(tuple(4 if dim == "B" else dim for dim in shape))
    return concrete_shapes


def run_mnist_example(
    config: MnistExampleConfig,
    *,
    create_model: ModelFactory,
    get_dataloaders: DataLoaderFactory,
    resolve_checkpoint_resume: CheckpointResolver,
    load_model: CheckpointLoader,
    train_model: TrainFn,
    visualize_results: VisualizeFn,
    test_onnx_model: OnnxEvalFn,
    to_onnx_fn: Callable[[nnx.Module, list[tuple[Any, ...]], Dict[str, Any]], onnx.ModelProto],
    allclose_fn: Callable[[nnx.Module, str, list[jax.Array], Dict[str, Any]], bool],
    save_onnx_model: Callable[[onnx.ModelProto, str | Path], None],
) -> None:
    config.validate()
    train_dataloader, test_dataloader = get_dataloaders(
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
        visualize_results(
            metrics_history,
            model,
            test_dataloader,
            start_epoch + config.training.num_epochs_to_train_now - 1,
            output_dir=str(config.artifact_dir()),
        )

    input_shapes = config.onnx.input_shapes
    input_params = config.onnx.input_params
    output_path = Path(config.onnx.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("Exporting model to ONNX...")
    onnx_model = to_onnx_fn(model, input_shapes, input_params)
    save_onnx_model(onnx_model, str(output_path))
    print(f"Model exported to {output_path}")

    concrete_shapes = _build_concrete_shapes(input_shapes)
    xs = [jax.random.normal(rng_key, shape) for shape in concrete_shapes]

    model.eval()
    result = allclose_fn(model, str(output_path), xs, input_params)
    print(f"ONNX allclose result: {result}")

    test_onnx_model(str(output_path), test_dataloader)
