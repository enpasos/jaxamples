# jaxamples
Examples build on JAX, Flax, nnx. We use jax2onnx to convert models to onnx format.

### **Examples**
* [`mnist_vit`](https://netron.app/?url=https://enpasos.github.io/jaxamples/onnx/mnist_vit_model.onnx) - MNIST classification using a vision transformer with convolutional embedding and a `cls_mean` classifier head.
* `mnist_dinov3` - MNIST classification using the DINOv3 Vision Transformer from `jax2onnx` plus a CLS classification head, configured with the same 700-epoch training budget and a denser 4x4 patch grid for a fairer comparison to `mnist_vit`.
* `mnist_cnn` - convolutional MNIST baseline trained through the same shared pipeline and with the same default augmentation as `mnist_vit` and `mnist_dinov3`, so architecture comparisons land in the same benchmark memory.
* `mnist_strong_cnn` - stronger residual CNN with LayerNorm, trained through the same default MNIST pipeline as the other examples so comparisons isolate the model architecture.

### **Run**
Install dependencies, train the model and export it to onnx format:
```  
poetry install
poetry run python jaxamples/mnist_vit.py
poetry run python jaxamples/mnist_dinov3.py
poetry run python jaxamples/mnist_cnn.py
poetry run python jaxamples/mnist_strong_cnn.py
```

Test exported models with ONNX Runtime:
```
poetry run python jaxamples/mnist_vit_run_onnx.py
poetry run python jaxamples/mnist_dinov3_run_onnx.py
poetry run python jaxamples/mnist_cnn_run_onnx.py
poetry run python jaxamples/mnist_strong_cnn_run_onnx.py
```

Quick overrides for both training scripts:
```
poetry run python jaxamples/mnist_vit.py --epochs 5 --batch-size 128
poetry run python jaxamples/mnist_dinov3.py --skip-training --checkpoint-dir ./data/dinov3_checkpoints
poetry run python jaxamples/mnist_vit.py --output-dir ./runs/mnist-vit
```

Plots and metrics go to `output/` by default. Exported ONNX models and their resolved configs go to `onnx/` by default. You can still override the export path explicitly with `--onnx-output`.

By default, local runs write training artifacts to `output/` and ONNX exports to `onnx/`.

Each training/export run also writes the resolved config next to the ONNX model, for example `onnx/mnist_vit_model_config.json`.

Every completed training run also appends a compact summary to `output/benchmark_memory.jsonl`, including the model name, augmentation fingerprint, best/final train and test accuracies, and pointers to the config and metrics artifacts. That makes it easier to compare `mnist_dinov3`, `mnist_vit`, `mnist_cnn`, and `mnist_strong_cnn` under identical default training conditions.

### **Quality checks**
The same checks are intended to run locally and in CI:
```
poetry check
poetry run python -m compileall jaxamples tests
poetry run ruff check jaxamples tests
poetry run pytest -q
```

To mirror the GitHub Actions CI job more closely, including `poetry install` and the CI Poetry environment variables, run:
```
bash scripts/run_ci_local.sh
```
