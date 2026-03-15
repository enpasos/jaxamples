from jaxamples.mnist_vit_run_onnx import main as run_mnist_onnx_main


def main(args=None):
    run_mnist_onnx_main(
        args=args,
        description="Test ONNX MNIST strong CNN model",
        default_onnx_model="onnx/mnist_strong_cnn_model.onnx",
    )


if __name__ == "__main__":
    main()
