from jaxamples.mnist_vit_run_onnx import main as run_mnist_onnx_main


def main(args=None):
    return run_mnist_onnx_main(
        args=args,
        description="Test ONNX MNIST DINOv3 model",
        default_onnx_model="output/mnist_dinov3_model.onnx",
    )


if __name__ == "__main__":
    main()
