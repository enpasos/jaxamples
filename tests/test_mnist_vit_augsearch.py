from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from torch.utils.data import DataLoader, TensorDataset

from jaxamples import mnist_training
from jaxamples import mnist_vit_augsearch
from jaxamples.mnist_config import MnistVitModelConfig


def create_small_model():
    model_config = MnistVitModelConfig(
        height=28,
        width=28,
        num_hiddens=64,
        num_layers=2,
        num_heads=4,
        mlp_dim=128,
        num_classes=10,
        embed_dims=[16, 32, 64],
        kernel_size=3,
        strides=[1, 2, 2],
        embedding_type="conv",
        embedding_dropout_rate=0.1,
        attention_dropout_rate=0.1,
        mlp_dropout_rate=0.1,
        head_hidden_dim=64,
        head_dropout_rate=0.1,
        pool_features="cls_mean",
    )
    return mnist_vit_augsearch.create_model(model_config, seed=0)


def get_single_batch_dataloaders():
    train_images = torch.zeros(4, 1, 28, 28, dtype=torch.float32)
    train_labels = torch.zeros(4, dtype=torch.int64)
    test_images = torch.ones(4, 1, 28, 28, dtype=torch.float32)
    test_labels = torch.ones(4, dtype=torch.int64)
    train_loader = DataLoader(
        TensorDataset(train_images, train_labels), batch_size=4, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(test_images, test_labels), batch_size=4, shuffle=False
    )
    return train_loader, test_loader


def get_dummy_dataloaders(batch_size: int):
    images = torch.ones(8, 1, 28, 28, dtype=torch.float32)
    labels = torch.arange(8, dtype=torch.int64) % 10
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, dataloader


def test_get_default_config_uses_separate_artifacts_and_search_defaults():
    config = mnist_vit_augsearch.get_default_config()

    assert config.training.checkpoint_dir.endswith("mnist_vit_augsearch_checkpoints")
    assert config.training.output_dir.endswith("output/mnist_vit_augsearch")
    assert config.onnx.model_name == "mnist_vit_augsearch_model"
    assert config.search.anchor_max_examples == 2048
    assert config.search.anchor_examples_per_class == 256
    assert config.search.invariance_threshold == pytest.approx(0.995)
    assert config.search.margin_retention_threshold == pytest.approx(0.9)
    assert config.search.strength_step == pytest.approx(0.2)


def test_build_parameter_scaled_augmentation_bootstraps_probability_axes():
    config = mnist_vit_augsearch.get_default_config()
    scaled = mnist_vit_augsearch.build_parameter_scaled_augmentation(
        config.training.augmentation,
        {
            "translation_probability": 0.3,
            "max_translation": 0.0,
            "scaling_probability": 0.4,
            "scale_span_x": 0.0,
            "scale_span_y": 0.0,
        },
        config.search,
    )

    assert scaled.enable_translation is True
    assert scaled.translation_probability == pytest.approx(0.3)
    assert scaled.max_translation > 0.0
    assert scaled.enable_scaling is True
    assert scaled.scaling_probability == pytest.approx(0.4)
    assert scaled.scale_min_x < 1.0
    assert scaled.scale_max_x > 1.0
    assert scaled.scale_min_y < 1.0
    assert scaled.scale_max_y > 1.0


def test_build_parameter_scaled_augmentation_expands_ranges_individually():
    config = mnist_vit_augsearch.get_default_config()
    scaled = mnist_vit_augsearch.build_parameter_scaled_augmentation(
        config.training.augmentation,
        {
            "translation_probability": 1.0,
            "max_translation": 1.0,
            "rotation_probability": 1.0,
            "max_rotation": 1.0,
            "elastic_probability": 1.0,
            "elastic_alpha": 1.0,
            "elastic_sigma": 1.0,
            "rect_erasing_probability": 1.0,
            "rect_erase_height": 1.0,
            "rect_erase_width": 1.0,
        },
        config.search,
    )

    assert scaled.max_translation >= 6.0
    assert scaled.max_rotation >= 18.0
    assert scaled.elastic_alpha >= 2.0
    assert scaled.elastic_sigma >= 2.0
    assert scaled.enable_rect_erasing is True
    assert scaled.rect_erase_height >= 4
    assert scaled.rect_erase_width >= 24


def test_build_parameter_scaled_augmentation_clamps_rect_erasing_to_image_shape():
    config = mnist_vit_augsearch.get_default_config()
    scaled = mnist_vit_augsearch.build_parameter_scaled_augmentation(
        config.training.augmentation,
        {
            "rect_erasing_probability": 1.0,
            "rect_erase_height": 1.0,
            "rect_erase_width": 1.0,
        },
        config.search,
        image_height=28,
        image_width=28,
    )

    assert scaled.enable_rect_erasing is True
    assert scaled.rect_erase_height <= 28
    assert scaled.rect_erase_width <= 28


def test_format_augmentation_summary_uses_actual_values():
    config = mnist_vit_augsearch.get_default_config()
    augmentation = mnist_vit_augsearch.build_parameter_scaled_augmentation(
        config.training.augmentation,
        {
            "scaling_probability": 1.0,
            "scale_span_y": 1.0,
            "max_rotation": 0.55,
            "rect_erasing_probability": 0.15,
            "rect_erase_height": 0.15,
        },
        config.search,
        image_height=28,
        image_width=28,
    )

    summary = mnist_vit_augsearch.format_augmentation_summary(augmentation)

    assert "scale[" in summary
    assert "p=100.0%" in summary
    assert "y=0.80..1.20" in summary
    assert "rotate[" in summary
    assert "max=9.9deg" in summary
    assert "erase[" in summary
    assert "h=" in summary


def test_format_augmentation_changes_uses_real_augmentation_values():
    config = mnist_vit_augsearch.get_default_config()
    before = mnist_vit_augsearch.build_parameter_scaled_augmentation(
        config.training.augmentation,
        {
            "scaling_probability": 0.719,
            "scale_span_y": 0.972,
            "max_rotation": 0.15,
            "rect_erasing_probability": 0.0,
            "rect_erase_height": 0.0,
        },
        config.search,
        image_height=28,
        image_width=28,
    )
    after = mnist_vit_augsearch.build_parameter_scaled_augmentation(
        config.training.augmentation,
        {
            "scaling_probability": 1.0,
            "scale_span_y": 1.0,
            "max_rotation": 0.55,
            "rect_erasing_probability": 0.15,
            "rect_erase_height": 0.15,
        },
        config.search,
        image_height=28,
        image_width=28,
    )

    changes = mnist_vit_augsearch.format_augmentation_changes(before, after)

    assert "scale: p 71.9%->100.0%" in changes
    assert "y 0.81..1.19->0.80..1.20" in changes
    assert "rotate: max 2.7deg->9.9deg" in changes
    assert "erase:" in changes


def test_augment_data_batch_clamps_oversized_rect_erasing_window():
    batch = {
        "image": jnp.ones((1, 28, 28, 1), dtype=jnp.float32),
        "label": jnp.array([0], dtype=jnp.int32),
    }
    augmentation_params = mnist_training.AugmentationParams(
        max_translation=0.0,
        scale_min_x=1.0,
        scale_max_x=1.0,
        scale_min_y=1.0,
        scale_max_y=1.0,
        max_rotation=0.0,
        elastic_alpha=0.0,
        elastic_sigma=1.0,
        enable_elastic=False,
        enable_rotation=False,
        enable_scaling=False,
        enable_translation=False,
        enable_rect_erasing=True,
        rect_erase_height=40,
        rect_erase_width=30,
        rect_erasing_probability=1.0,
    )

    search_augment_batch = getattr(
        mnist_training.augment_data_batch,
        "__wrapped__",
        mnist_training.augment_data_batch,
    )
    augmented = search_augment_batch(batch, jax.random.PRNGKey(0), augmentation_params)

    assert augmented["image"].shape == batch["image"].shape
    assert jnp.all(augmented["image"] == 0.0)


def test_search_parameter_frontier_expands_until_retention_boundary():
    search_config = mnist_vit_augsearch.AugmentationSearchConfig(
        invariance_threshold=0.95,
        margin_retention_threshold=0.9,
        strength_step=0.25,
        min_strength_step=0.05,
    )

    frontier, evaluation = mnist_vit_augsearch.search_parameter_frontier(
        0.0,
        lambda value: mnist_vit_augsearch.CandidateEvaluation(
            retention=0.97 if value <= 0.6 else 0.90,
            mean_margin=1.0,
            margin_retention=0.95,
        ),
        search_config,
    )

    assert 0.55 <= frontier <= 0.65
    assert evaluation.retention >= 0.95


def test_search_parameter_frontier_can_shrink_existing_value():
    search_config = mnist_vit_augsearch.AugmentationSearchConfig(
        invariance_threshold=0.95,
        margin_retention_threshold=0.9,
        strength_step=0.25,
        min_strength_step=0.05,
    )

    frontier, evaluation = mnist_vit_augsearch.search_parameter_frontier(
        0.8,
        lambda value: mnist_vit_augsearch.CandidateEvaluation(
            retention=0.97 if value <= 0.35 else 0.80,
            mean_margin=1.0,
            margin_retention=0.95,
        ),
        search_config,
    )

    assert 0.30 <= frontier <= 0.40
    assert evaluation.retention >= 0.95


def test_search_parameter_frontier_respects_margin_retention_threshold():
    search_config = mnist_vit_augsearch.AugmentationSearchConfig(
        invariance_threshold=0.95,
        margin_retention_threshold=0.9,
        strength_step=0.25,
        min_strength_step=0.05,
    )

    frontier, evaluation = mnist_vit_augsearch.search_parameter_frontier(
        0.0,
        lambda value: mnist_vit_augsearch.CandidateEvaluation(
            retention=0.99,
            mean_margin=1.0,
            margin_retention=0.95 if value <= 0.4 else 0.75,
        ),
        search_config,
    )

    assert 0.35 <= frontier <= 0.45
    assert evaluation.margin_retention >= 0.9


def test_search_parameter_candidates_can_choose_interior_sigma_value():
    search_config = mnist_vit_augsearch.AugmentationSearchConfig(
        invariance_threshold=0.95,
        margin_retention_threshold=0.9,
        strength_step=0.25,
        min_strength_step=0.05,
    )
    candidate_values = [0.0, 0.2, 0.35, 0.6, 1.0]

    frontier, evaluation = mnist_vit_augsearch.search_parameter_candidates(
        0.0,
        candidate_values,
        lambda value: mnist_vit_augsearch.CandidateEvaluation(
            retention=0.95 + abs(value - 0.35),
            mean_margin=1.0,
            margin_retention=0.90 + abs(value - 0.35),
        ),
        search_config,
    )

    assert frontier == pytest.approx(0.35)
    assert evaluation.retention == pytest.approx(0.95)
    assert evaluation.margin_retention == pytest.approx(0.90)


def test_effective_search_config_anneals_steps_from_train_plateau_state():
    search_config = mnist_vit_augsearch.AugmentationSearchConfig(
        strength_step=0.2,
        min_strength_step=0.05,
        annealed_strength_step=0.04,
        annealed_min_strength_step=0.01,
        train_plateau_patience=10,
        anneal_after_plateau_fraction=0.5,
    )
    state = replace(
        mnist_vit_augsearch.default_search_state(search_config),
        epochs_since_train_improvement=8,
    )

    effective = mnist_vit_augsearch._effective_search_config(search_config, state)

    assert 0.04 < effective.strength_step < 0.2
    assert 0.01 < effective.min_strength_step < 0.05


def test_is_scheduled_search_epoch_starts_at_epoch_one():
    search_config = mnist_vit_augsearch.AugmentationSearchConfig(
        search_every_n_epochs=3,
    )

    assert mnist_vit_augsearch._is_scheduled_search_epoch(1, search_config) is True
    assert mnist_vit_augsearch._is_scheduled_search_epoch(2, search_config) is False
    assert mnist_vit_augsearch._is_scheduled_search_epoch(3, search_config) is False
    assert mnist_vit_augsearch._is_scheduled_search_epoch(4, search_config) is True


def test_should_refresh_artifacts_only_on_interval_and_final_epoch():
    assert (
        mnist_vit_augsearch._should_refresh_artifacts(
            1,
            final_epoch=6,
            refresh_interval=5,
        )
        is False
    )
    assert (
        mnist_vit_augsearch._should_refresh_artifacts(
            5,
            final_epoch=6,
            refresh_interval=5,
        )
        is True
    )
    assert (
        mnist_vit_augsearch._should_refresh_artifacts(
            6,
            final_epoch=6,
            refresh_interval=5,
        )
        is True
    )


def test_update_search_train_plateau_state_freezes_from_clean_train_accuracy():
    search_config = mnist_vit_augsearch.AugmentationSearchConfig(
        train_plateau_window=2,
        train_plateau_patience=2,
        train_plateau_min_improvement=1e-4,
    )
    state = mnist_vit_augsearch.default_search_state(search_config)

    state, metric, message = mnist_vit_augsearch.update_search_train_plateau_state(
        state,
        [0.99],
        epoch=0,
        search_config=search_config,
    )
    assert metric == pytest.approx(0.99)
    assert message is None
    assert state.best_train_accuracy_metric == pytest.approx(0.99)
    assert state.epochs_since_train_improvement == 0
    assert state.search_frozen is False

    state, metric, message = mnist_vit_augsearch.update_search_train_plateau_state(
        state,
        [0.99, 0.99],
        epoch=1,
        search_config=search_config,
    )
    assert metric == pytest.approx(0.99)
    assert message is None
    assert state.epochs_since_train_improvement == 1
    assert state.search_frozen is False

    state, metric, message = mnist_vit_augsearch.update_search_train_plateau_state(
        state,
        [0.99, 0.99, 0.99],
        epoch=2,
        search_config=search_config,
    )
    assert metric == pytest.approx(0.99)
    assert message is not None
    assert state.epochs_since_train_improvement == 2
    assert state.search_frozen is True
    assert state.frozen_epoch == 2
    assert state.last_event == "search_train_plateau_frozen"


def test_search_augmentation_parameters_skips_when_search_is_frozen(monkeypatch):
    config = mnist_vit_augsearch.get_default_config()
    state = replace(
        mnist_vit_augsearch.default_search_state(config.search),
        search_frozen=True,
        frozen_epoch=7,
    )

    monkeypatch.setattr(
        mnist_vit_augsearch,
        "_collect_anchor_set",
        lambda *args, **kwargs: pytest.fail("frozen search must not collect anchors"),
    )

    next_state, anchor, updates = mnist_vit_augsearch.search_augmentation_parameters(
        object(),
        object(),
        config.training.augmentation,
        state,
        config,
        epoch=8,
    )

    assert anchor is None
    assert updates == []
    assert next_state.search_frozen is True
    assert next_state.last_event == "search_frozen"


def test_apply_search_overrides_updates_search_interval():
    config = mnist_vit_augsearch.get_default_config()
    args = mnist_vit_augsearch.parse_augsearch_args(
        ["--search-every-n-epochs", "4"],
        description="test",
        default_onnx_output="out.onnx",
    )

    updated = mnist_vit_augsearch._apply_search_overrides(config, args)

    assert updated.search.search_every_n_epochs == 4


def test_collect_anchor_set_balances_classes(monkeypatch):
    model = create_small_model()
    images = torch.zeros(6, 1, 28, 28, dtype=torch.float32)
    labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)
    dataloader = DataLoader(
        TensorDataset(images, labels),
        batch_size=6,
        shuffle=False,
    )

    def fake_logits_step(_model, batch):
        logits = jnp.full((batch["label"].shape[0], 10), -2.0)
        return logits.at[jnp.arange(batch["label"].shape[0]), batch["label"]].set(2.0)

    monkeypatch.setattr(mnist_vit_augsearch, "_logits_step", fake_logits_step)

    anchor = mnist_vit_augsearch._collect_anchor_set(
        model,
        dataloader,
        max_examples=4,
        max_examples_per_class=2,
        num_classes=2,
        min_margin=0.0,
    )

    assert anchor is not None
    assert int(anchor.labels.shape[0]) == 4
    assert anchor.class_counts == [2, 2]


def test_search_augmentation_parameters_updates_probability_then_magnitude(
    monkeypatch,
):
    model = create_small_model()
    config = mnist_vit_augsearch.get_default_config()
    config.search.invariance_threshold = 0.95
    config.search.strength_step = 0.2
    config.search.min_strength_step = 0.05
    state = mnist_vit_augsearch.default_search_state(config.search)
    reference_augmentation = config.training.augmentation

    monkeypatch.setattr(
        mnist_vit_augsearch,
        "_collect_anchor_set",
        lambda *args, **kwargs: mnist_vit_augsearch.AnchorSet(
            images=jnp.zeros((4, 28, 28, 1), dtype=jnp.float32),
            labels=jnp.zeros((4,), dtype=jnp.int32),
            class_counts=[4] + [0] * 9,
            mean_margin=1.25,
        ),
    )

    thresholds = {
        0: 0.65,  # translation_probability
        1: 0.40,  # max_translation
    }

    def fake_evaluate(*args, parameter_index, normalized_value, **kwargs):
        threshold = thresholds.get(parameter_index, 0.0)
        retention = 0.97 if normalized_value <= threshold + 1e-9 else 0.80
        return mnist_vit_augsearch.CandidateEvaluation(
            retention=retention,
            mean_margin=1.0,
            margin_retention=0.95,
        )

    monkeypatch.setattr(
        mnist_vit_augsearch,
        "evaluate_invariance_candidate",
        fake_evaluate,
    )

    next_state, anchor, updates = mnist_vit_augsearch.search_augmentation_parameters(
        model,
        object(),
        reference_augmentation,
        state,
        config,
        epoch=3,
    )

    update_names = [update.parameter for update in updates]
    assert update_names[:2] == ["translation_probability", "max_translation"]
    assert 0.55 <= next_state.parameter_values["translation_probability"] <= 0.70
    assert 0.35 <= next_state.parameter_values["max_translation"] <= 0.45
    assert anchor is not None
    assert next_state.last_anchor_size == 4
    assert next_state.last_anchor_num_classes == 1
    assert next_state.last_anchor_min_class_count == 4
    assert next_state.last_margin_retention == pytest.approx(0.95)
    assert next_state.last_event == "search_updated"


def test_load_search_state_resets_legacy_scalar_state(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    legacy_payload = {
        "phase": "baseline",
        "baseline_strength": 0.85,
        "active_strength": 0.85,
        "candidate_strength": None,
        "failed_upper_strength": 0.9,
        "step_size": 0.05,
        "probe_epochs_used": 0,
        "probe_start_epoch": None,
        "search_enabled": False,
        "last_event": "search_disabled",
    }
    (checkpoint_dir / "augmentation_search_state.json").write_text(
        mnist_vit_augsearch.json.dumps(legacy_payload),
        encoding="utf-8",
    )

    restored_state = mnist_vit_augsearch.load_search_state(
        str(checkpoint_dir),
        mnist_vit_augsearch.AugmentationSearchConfig(),
    )

    assert restored_state.last_event == "reset_search_state"
    assert restored_state.parameter_values["translation_probability"] == pytest.approx(0.0)
    assert restored_state.last_updated_parameters == []


def test_create_and_restore_training_snapshot_restores_model_and_optimizer():
    model = create_small_model()
    optimizer = mnist_vit_augsearch.create_optimizer(model, 1e-3, 1e-4)
    rng_key = jax.random.PRNGKey(7)

    original_model_state = nnx.state(model, nnx.RngKey, ...)
    original_optimizer_state = nnx.state(optimizer)
    snapshot = mnist_vit_augsearch.create_training_snapshot(model, optimizer, rng_key)

    model.head_out.bias[...] = model.head_out.bias[...] + 1.0
    optimizer.step[...] = optimizer.step[...] + jnp.array(
        3, dtype=optimizer.step[...].dtype
    )

    restored_rng = mnist_vit_augsearch.restore_training_snapshot(
        model, optimizer, snapshot
    )
    restored_model_state = nnx.state(model, nnx.RngKey, ...)
    restored_optimizer_state = nnx.state(optimizer)

    model_equal = jax.tree_util.tree_map(
        lambda x, y: jnp.allclose(x, y, atol=1e-6),
        original_model_state,
        restored_model_state,
    )
    optimizer_equal = jax.tree_util.tree_map(
        lambda x, y: jnp.allclose(x, y, atol=1e-6),
        original_optimizer_state,
        restored_optimizer_state,
    )

    assert jax.tree_util.tree_all(model_equal)
    assert jax.tree_util.tree_all(optimizer_equal)
    assert np.array_equal(
        np.asarray(jax.random.key_data(restored_rng)),
        np.asarray(jax.random.key_data(rng_key)),
    )


def test_train_model_continues_across_stable_search_epochs(monkeypatch, tmp_path):
    class FakeMetrics:
        def __init__(self):
            self.current = {"loss": jnp.array(0.0), "accuracy": jnp.array(0.0)}

        def reset(self):
            self.current = {"loss": jnp.array(0.0), "accuracy": jnp.array(0.0)}

        def compute(self):
            return self.current

    model = create_small_model()
    train_loader, test_loader = get_single_batch_dataloaders()
    config = mnist_vit_augsearch.get_default_config()
    config.training.num_epochs_to_train_now = 2
    config.training.warmup_epochs = 0
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")
    config.training.output_dir = str(tmp_path / "artifacts")
    config.training.batch_size = 4

    search_calls = []

    def fake_search(_model, _train_eval_loader, _reference_augmentation, state, _config, epoch):
        search_calls.append(epoch)
        updated_state = replace(
            state,
            parameter_values={
                **state.parameter_values,
                "translation_probability": 0.4,
            },
            last_anchor_size=4,
            last_anchor_num_classes=1,
            last_anchor_min_class_count=4,
            last_anchor_mean_margin=1.0,
            last_retention=0.99,
            last_margin_retention=0.96,
            last_updated_parameters=["translation_probability"] if epoch == 0 else [],
            last_event="search_updated" if epoch == 0 else "search_stable",
        )
        anchor = mnist_vit_augsearch.AnchorSet(
            images=jnp.zeros((4, 28, 28, 1), dtype=jnp.float32),
            labels=jnp.zeros((4,), dtype=jnp.int32),
            class_counts=[4] + [0] * 9,
            mean_margin=1.0,
        )
        updates = (
            [
                mnist_vit_augsearch.ParameterSearchUpdate(
                    parameter="translation_probability",
                    old_value=0.0,
                    new_value=0.4,
                    retention=0.99,
                    mean_margin=1.0,
                    margin_retention=0.96,
                )
            ]
            if epoch == 0
            else []
        )
        return updated_state, anchor, updates

    def fake_train_step(_model, _optimizer, metrics, _batch, _learning_rate, _weight_decay):
        metrics.current = {"loss": jnp.array(1.0), "accuracy": jnp.array(0.75)}

    def fake_eval_step(_model, metrics, _batch):
        metrics.current = {"loss": jnp.array(0.0), "accuracy": jnp.array(1.0)}

    monkeypatch.setattr(mnist_vit_augsearch, "search_augmentation_parameters", fake_search)
    monkeypatch.setattr(mnist_vit_augsearch, "augment_data_batch", lambda batch, *_args: batch)
    monkeypatch.setattr(mnist_vit_augsearch, "train_step", fake_train_step)
    monkeypatch.setattr(mnist_vit_augsearch, "eval_step", fake_eval_step)
    monkeypatch.setattr(
        mnist_vit_augsearch, "visualize_augmented_images", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        mnist_vit_augsearch,
        "visualize_incorrect_classifications",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(mnist_vit_augsearch, "save_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mnist_vit_augsearch,
        "load_and_plot_test_accuracy_metrics",
        lambda *args, **kwargs: None,
    )

    metrics_history = mnist_vit_augsearch.train_model(
        model,
        0,
        FakeMetrics(),
        config,
        train_loader,
        test_loader,
        jax.random.PRNGKey(0),
    )

    assert search_calls == [0, 1]
    assert metrics_history["search_event"] == ["search_updated", "search_stable"]
    assert metrics_history["search_num_updates"] == [1, 0]
    assert metrics_history["search_margin_retention"] == [0.96, 0.96]
    assert len(metrics_history["test_accuracy"]) == 2

    saved_state = mnist_vit_augsearch.load_search_state(
        config.training.checkpoint_dir,
        config.search,
    )
    assert saved_state.last_event == "search_stable"
    assert saved_state.parameter_values["translation_probability"] == pytest.approx(0.4)


def test_main_skips_visualization_when_no_epochs_were_trained(monkeypatch, tmp_path):
    calls = {}
    train_dataloader, test_dataloader = get_dummy_dataloaders(batch_size=8)

    config = mnist_vit_augsearch.get_default_config()
    config.training.num_epochs_to_train_now = 5
    config.training.data_dir = str(tmp_path / "data")
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")
    config.training.output_dir = str(tmp_path / "output")
    config.onnx.output_path = str(tmp_path / "docs" / "mnist_vit_augsearch_model.onnx")

    monkeypatch.setattr(mnist_vit_augsearch, "get_default_config", lambda: config)
    monkeypatch.setattr(
        mnist_vit_augsearch,
        "get_dataset_torch_dataloaders",
        lambda batch_size, data_dir: (train_dataloader, test_dataloader),
    )
    monkeypatch.setattr(
        mnist_vit_augsearch, "resolve_checkpoint_resume", lambda ckpt_dir: (None, 0)
    )
    monkeypatch.setattr(
        mnist_vit_augsearch,
        "train_model",
        lambda *args, **kwargs: mnist_vit_augsearch._initialize_metrics_history(),
    )
    monkeypatch.setattr(
        mnist_vit_augsearch,
        "visualize_results",
        lambda *args, **kwargs: calls.setdefault("visualize_results", True),
    )
    monkeypatch.setattr(
        mnist_vit_augsearch, "to_onnx", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        mnist_vit_augsearch.onnx, "save_model", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(mnist_vit_augsearch, "allclose", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        mnist_vit_augsearch,
        "test_onnx_model",
        lambda *args, **kwargs: calls.setdefault("test_onnx_model", True),
    )

    mnist_vit_augsearch.main(
        [
            "--epochs",
            "5",
            "--batch-size",
            "8",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--onnx-output",
            str(tmp_path / "docs" / "mnist_vit_augsearch_model.onnx"),
        ]
    )

    assert "visualize_results" not in calls
    benchmark_memory_path = tmp_path / "output" / "benchmark_memory.jsonl"
    assert benchmark_memory_path.exists()


def test_main_applies_cli_overrides_and_writes_benchmark(monkeypatch, tmp_path):
    calls = {}
    train_dataloader, test_dataloader = get_dummy_dataloaders(batch_size=8)

    config = mnist_vit_augsearch.get_default_config()
    config.training.num_epochs_to_train_now = 1
    config.training.data_dir = str(tmp_path / "data")
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")
    config.training.output_dir = str(tmp_path / "output")
    config.onnx.output_path = str(tmp_path / "docs" / "mnist_vit_augsearch_model.onnx")

    monkeypatch.setattr(mnist_vit_augsearch, "get_default_config", lambda: config)

    def fake_get_dataset_torch_dataloaders(batch_size, data_dir):
        calls["loader_args"] = {"batch_size": batch_size, "data_dir": data_dir}
        return train_dataloader, test_dataloader

    monkeypatch.setattr(
        mnist_vit_augsearch,
        "get_dataset_torch_dataloaders",
        fake_get_dataset_torch_dataloaders,
    )
    monkeypatch.setattr(
        mnist_vit_augsearch, "resolve_checkpoint_resume", lambda ckpt_dir: (None, 0)
    )

    def fake_train_model(
        model, start_epoch, metrics, config, train_loader, test_loader, rng_key
    ):
        calls["train_model"] = {
            "start_epoch": start_epoch,
            "anchor_max_examples": config.search.anchor_max_examples,
            "anchor_examples_per_class": config.search.anchor_examples_per_class,
            "invariance_threshold": config.search.invariance_threshold,
            "margin_retention_threshold": config.search.margin_retention_threshold,
            "strength_step": config.search.strength_step,
        }
        history = mnist_vit_augsearch._initialize_metrics_history()
        history["train_loss"] = [1.0]
        history["test_loss"] = [1.0]
        history["train_accuracy"] = [0.5]
        history["test_accuracy"] = [0.5]
        history["train_accuracy_mean"] = [0.0]
        history["train_accuracy_spread"] = [0.0]
        history["test_accuracy_mean"] = [0.0]
        history["test_accuracy_spread"] = [0.0]
        history["train_error_count"] = [25]
        history["search_anchor_size"] = [128]
        history["search_anchor_num_classes"] = [10]
        history["search_anchor_min_class_count"] = [12]
        history["search_anchor_mean_margin"] = [1.2]
        history["search_retention"] = [0.996]
        history["search_margin_retention"] = [0.94]
        history["search_num_updates"] = [2]
        history["search_updated_parameters"] = [
            "translation_probability 0.000->0.600; max_translation 0.000->0.400"
        ]
        history["search_event"] = ["search_updated"]
        return history

    def fake_visualize_results(metrics_history, model, test_loader, epoch, output_dir="output"):
        calls["visualize_results"] = {"epoch": epoch, "output_dir": output_dir}

    def fake_to_onnx(model, input_shapes, input_params):
        calls["to_onnx"] = {"input_shapes": input_shapes, "input_params": input_params}
        return object()

    def fake_save_model(model_proto, output_path):
        calls["save_model"] = output_path

    def fake_test_onnx_model(output_path, test_loader):
        calls["test_onnx_model"] = {"output_path": output_path, "batches": len(test_loader)}

    monkeypatch.setattr(mnist_vit_augsearch, "train_model", fake_train_model)
    monkeypatch.setattr(mnist_vit_augsearch, "visualize_results", fake_visualize_results)
    monkeypatch.setattr(mnist_vit_augsearch, "to_onnx", fake_to_onnx)
    monkeypatch.setattr(mnist_vit_augsearch.onnx, "save_model", fake_save_model)
    monkeypatch.setattr(mnist_vit_augsearch, "allclose", lambda *args, **kwargs: True)
    monkeypatch.setattr(mnist_vit_augsearch, "test_onnx_model", fake_test_onnx_model)

    mnist_vit_augsearch.main(
        [
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--onnx-output",
            str(tmp_path / "docs" / "mnist_vit_augsearch_model.onnx"),
            "--anchor-max-examples",
            "128",
            "--anchor-examples-per-class",
            "16",
            "--invariance-threshold",
            "0.97",
            "--margin-retention-threshold",
            "0.88",
            "--strength-step",
            "0.3",
        ]
    )

    assert calls["loader_args"] == {
        "batch_size": 8,
        "data_dir": str(tmp_path / "data"),
    }
    assert calls["train_model"]["start_epoch"] == 0
    assert calls["train_model"]["anchor_max_examples"] == 128
    assert calls["train_model"]["anchor_examples_per_class"] == 16
    assert calls["train_model"]["invariance_threshold"] == pytest.approx(0.97)
    assert calls["train_model"]["margin_retention_threshold"] == pytest.approx(0.88)
    assert calls["train_model"]["strength_step"] == pytest.approx(0.3)
    assert calls["to_onnx"]["input_shapes"] == [("B", 28, 28, 1)]
    assert calls["to_onnx"]["input_params"] == {"deterministic": True}
    assert calls["save_model"] == str(tmp_path / "docs" / "mnist_vit_augsearch_model.onnx")
    assert Path(tmp_path / "docs" / "mnist_vit_augsearch_model_config.json").exists()

    benchmark_memory_path = tmp_path / "output" / "benchmark_memory.jsonl"
    assert benchmark_memory_path.exists()
    content = benchmark_memory_path.read_text(encoding="utf-8")
    assert '"augmentation_strategy":"frozen_model_invariance_search"' in content
    assert '"anchor_max_examples":128' in content
    assert '"anchor_examples_per_class":16' in content
    assert '"invariance_threshold":0.97' in content
    assert '"margin_retention_threshold":0.88' in content
