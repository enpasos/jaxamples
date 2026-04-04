import jax.numpy as jnp

from jaxamples import mnist_vit_xsa
from jaxamples import mnist_vit_xsa_augsearch


def test_mnist_vit_xsa_default_config_uses_separate_artifacts():
    config = mnist_vit_xsa.get_default_config()

    assert config.training.checkpoint_dir.endswith("mnist_vit_xsa_cls_mean_checkpoints")
    assert config.training.output_dir.endswith("output/mnist_vit_xsa")
    assert config.onnx.model_name == "mnist_vit_xsa_model"
    assert config.onnx.output_path.endswith("onnx/mnist_vit_xsa_model.onnx")


def test_mnist_vit_xsa_create_model_matches_vit_shape_contract():
    config = mnist_vit_xsa.get_default_config()
    model = mnist_vit_xsa.create_model(config.model, seed=0)

    logits = model(jnp.ones((2, 28, 28, 1), dtype=jnp.float32), deterministic=True)

    assert logits.shape == (2, 10)
    assert jnp.isfinite(logits).all()


def test_mnist_vit_xsa_augsearch_default_config_uses_separate_artifacts():
    config = mnist_vit_xsa_augsearch.get_default_config()
    baseline = mnist_vit_xsa_augsearch.mnist_vit_augsearch_lib.get_default_config()

    assert config.training.checkpoint_dir.endswith(
        "mnist_vit_xsa_augsearch_checkpoints"
    )
    assert config.training.output_dir.endswith("output/mnist_vit_xsa_augsearch")
    assert config.onnx.model_name == "mnist_vit_xsa_augsearch_model"
    assert config.onnx.output_path.endswith("onnx/mnist_vit_xsa_augsearch_model.onnx")
    assert config.search.to_dict() == baseline.search.to_dict()
