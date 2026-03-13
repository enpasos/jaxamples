# jaxamples
Examples build on JAX, Flax, nnx. We use jax2onnx to convert models to onnx format.

### **Examples**
* [`mnist_vit`](https://netron.app/?url=https://enpasos.github.io/jaxamples/mnist_vit_model.onnx) - MNIST classification using a vision transformer with convolutional embedding. 
* `mnist_dinov3` - MNIST classification using the DINOv3 Vision Transformer from `jax2onnx` plus a CLS classification head, configured with the same 500-epoch training budget and a denser 4x4 patch grid for a fairer comparison to `mnist_vit`.

### **Run**
Install dependencies, train the model and export it to onnx format:
```  
poetry install
poetry run python jaxamples/mnist_vit.py
poetry run python jaxamples/mnist_dinov3.py
```

Test exported models with ONNX Runtime:
```
poetry run python jaxamples/mnist_vit_run_onnx.py
poetry run python jaxamples/mnist_dinov3_run_onnx.py
```

Quick overrides for both training scripts:
```
poetry run python jaxamples/mnist_vit.py --epochs 5 --batch-size 128
poetry run python jaxamples/mnist_dinov3.py --skip-training --checkpoint-dir ./data/dinov3_checkpoints
poetry run python jaxamples/mnist_vit.py --output-dir ./runs/mnist-vit
```

Plots and metrics go to `output/` by default. You can force a dedicated run directory with `--output-dir`; unless `--onnx-output` is set explicitly, the ONNX model and resolved config are written there too.

By default, local runs now write their generated artifacts to `output/`.

Each training/export run also writes the resolved config next to the ONNX model, for example `output/mnist_vit_model_config.json`.

### **Quality checks**
The same checks are intended to run locally and in CI:
```
poetry check
poetry run python -m compileall jaxamples tests
poetry run ruff check jaxamples tests
poetry run pytest -q
```
