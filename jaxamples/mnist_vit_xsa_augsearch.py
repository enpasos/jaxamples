import os
from pathlib import Path
from typing import Sequence

import jax
import onnx
from flax import nnx
from jax2onnx import allclose, to_onnx

from jaxamples import mnist_vit_augsearch as mnist_vit_augsearch_lib
from jaxamples import mnist_vit_xsa as mnist_vit_xsa_lib


get_dataset_torch_dataloaders = mnist_vit_augsearch_lib.get_dataset_torch_dataloaders
resolve_checkpoint_resume = mnist_vit_augsearch_lib.resolve_checkpoint_resume
load_model = mnist_vit_augsearch_lib.load_model
test_onnx_model = mnist_vit_augsearch_lib.test_onnx_model
visualize_results = mnist_vit_augsearch_lib.visualize_results
train_model = mnist_vit_augsearch_lib.train_model
load_search_state = mnist_vit_augsearch_lib.load_search_state
_append_augsearch_benchmark_record = (
    mnist_vit_augsearch_lib._append_augsearch_benchmark_record
)

def get_default_config() -> mnist_vit_augsearch_lib.MnistVitAugSearchConfig:
    base_config = mnist_vit_xsa_lib.get_default_config()
    default_output_dir = os.path.abspath("./output/mnist_vit_xsa_augsearch/")
    default_onnx_dir = os.path.abspath("./onnx/")
    base_config.training.checkpoint_dir = os.path.abspath(
        "./data/mnist_vit_xsa_augsearch_checkpoints/"
    )
    base_config.training.output_dir = default_output_dir
    base_config.onnx.model_name = "mnist_vit_xsa_augsearch_model"
    base_config.onnx.output_path = os.path.join(
        default_onnx_dir,
        "mnist_vit_xsa_augsearch_model.onnx",
    )
    config = mnist_vit_augsearch_lib.MnistVitAugSearchConfig(
        seed=base_config.seed,
        training=base_config.training,
        model=base_config.model,
        onnx=base_config.onnx,
        search=mnist_vit_augsearch_lib.AugmentationSearchConfig(),
    )
    return config


def create_model(model_config, seed: int) -> nnx.Module:
    return mnist_vit_xsa_lib.create_model(model_config, seed)


def main(args: Sequence[str] | None = None) -> None:
    cli_args = mnist_vit_augsearch_lib.parse_augsearch_args(
        args,
        description="Train and export the MNIST ViT XSA augmentation-search example",
        default_onnx_output="onnx/mnist_vit_xsa_augsearch_model.onnx",
    )
    config = mnist_vit_augsearch_lib.apply_common_overrides(get_default_config(), cli_args)
    config = mnist_vit_augsearch_lib._apply_search_overrides(config, cli_args)

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
            config,
            metrics_history,
            final_search_state,
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
