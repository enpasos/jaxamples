# jaxamples
Examples build on JAX, Flax, nnx. We use jax2onnx to convert models to onnx format.

### **Examples**
* [`mnist_vit`](https://netron.app/?url=https://enpasos.github.io/jaxamples/onnx/mnist_vit_model.onnx) - MNIST classification using a vision transformer with convolutional embedding. 

### **Run**
Install dependencies, train the model and export it to onnx format:
```  
poetry install
poetry run python jaxamples/mnist_vit.py
``` 
