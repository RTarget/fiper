"""Two-epoch CPU pipeline smoke. No test evaluation or performance claims."""

import json
import os
import subprocess
import math
from pathlib import Path

import numpy as np
import torch

from datasets.tac_fiper_a0 import ROLE_INDICES, make_a0_dataloader
from datasets.tac_fiper_a0_normalization import A0FeatureNormalizer
from evaluation.temporal_models import ObservationActionTemporalModel


def epoch_loss(model, loader, normalizer, optimizer=None):
    """Sample-weighted BCE; validation cannot update parameters."""
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for raw in loader:
            batch = normalizer.transform(raw)
            device = next(model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch["obs_embeddings"], batch["action_preds"], batch["padding_mask"])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["target"])
            if not torch.isfinite(loss).item():
                raise ValueError("non-finite BCE loss")
            if optimizer is not None:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                if not torch.isfinite(norm).item():
                    raise ValueError("non-finite gradient")
                optimizer.step()
            n = len(logits)
            total += loss.item() * n
            count += n
    if not count:
        raise ValueError("empty training/selection loader")
    return total / count


def rollout_threshold(rows, expected_indices):
    """ct_quantile: ordinary linear q=.9 quantile of rollout maxima."""
    maxima = {}
    seen = set()
    for rollout, timestep, score in rows:
        key = (rollout, timestep)
        if rollout not in expected_indices or key in seen or not math.isfinite(score):
            raise ValueError("invalid, duplicate, or non-finite threshold score")
        seen.add(key)
        maxima[rollout] = max(maxima.get(rollout, -math.inf), score)
    if set(maxima) != set(expected_indices):
        raise ValueError("threshold scores must cover every threshold rollout")
    values = np.asarray([maxima[i] for i in expected_indices], dtype=np.float64)
    return float(np.quantile(values, 0.9)), maxima


def run_cpu_smoke(source, output_dir, provenance, *, gpu_uuid=None):
    """Write artifacts into a NEW directory; fixed config, no hyperparameter search."""
    device = torch.device("cpu")
    gpu_info = None
    if gpu_uuid is not None:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid or not gpu_uuid.startswith("GPU-"):
            raise ValueError("CUDA_VISIBLE_DEVICES must equal the explicitly approved GPU UUID")
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        if any(line.split(",")[0].strip() == gpu_uuid for line in processes.splitlines()):
            raise RuntimeError("Selected GPU has a compute process; retry only when available")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one CUDA device must be visible")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        gpu_info = dict(uuid=gpu_uuid, name=torch.cuda.get_device_name(0))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    seed = 20260909
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    train = make_a0_dataloader(source, "representation", task="sorting", batch_size=32, seed=seed)
    selection = make_a0_dataloader(source, "selection", task="sorting", batch_size=32, seed=seed)
    normalizer = A0FeatureNormalizer.fit(train.dataset)
    config = dict(obs_dim=128, action_horizon=8, action_dim=6,
                  d_model=32, nhead=4, num_layers=1, dim_feedforward=64, dropout=0.0)
    model = ObservationActionTemporalModel(**config).to(device)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history, best_loss, best_epoch = [], math.inf, None
    checkpoint = output / "best.pt"
    for epoch in (1, 2):
        training_loss = epoch_loss(model, train, normalizer, optimizer)
        validation_loss = epoch_loss(model, selection, normalizer)
        history.append(dict(epoch=epoch, train_loss=training_loss, selection_loss=validation_loss))
        if validation_loss < best_loss:  # ties retain the earlier epoch
            best_loss, best_epoch = validation_loss, epoch
            torch.save(dict(model=model.state_dict(), model_config=config,
                            normalization=normalizer.state_dict(), epoch=epoch,
                            selection_loss=validation_loss, seed=seed), checkpoint)
        print(f"epoch={epoch}: train_loss={training_loss:.6f}, selection_loss={validation_loss:.6f}", flush=True)
    if not any(not torch.equal(initial[k], v) for k, v in model.state_dict().items()):
        raise ValueError("training did not change any parameters")

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"])
    model.eval()
    restored = ObservationActionTemporalModel(**saved["model_config"]).to(device).eval()
    restored.load_state_dict(saved["model"])
    restored_norm = A0FeatureNormalizer.from_state_dict(saved["normalization"])
    reload_loss = epoch_loss(restored, selection, restored_norm)
    if not math.isclose(reload_loss, best_loss, rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("reloaded checkpoint selection loss differs")
    raw = next(iter(selection))
    left, right = normalizer.transform(raw), restored_norm.transform(raw)
    left = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in left.items()}
    right = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in right.items()}
    with torch.no_grad():
        expected = model(left["obs_embeddings"], left["action_preds"], left["padding_mask"])
        actual = restored(right["obs_embeddings"], right["action_preds"], right["padding_mask"])
    if not torch.equal(expected, actual):
        raise ValueError("checkpoint/normalization reload outputs differ")

    # Threshold samples are constructed/scored only AFTER checkpoint selection.
    threshold = make_a0_dataloader(source, "threshold", task="sorting", batch_size=32, seed=seed)
    rows = []
    with torch.no_grad():
        for raw in threshold:
            batch = restored_norm.transform(raw)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = restored(batch["obs_embeddings"], batch["action_preds"], batch["padding_mask"])
            scores = torch.nn.functional.softplus(-logits)
            rows.extend((int(i), int(t), float(s)) for i, t, s in
                        zip(batch["rollout_index"], batch["timestep"], scores))
    if len(rows) != len(threshold.dataset):
        raise ValueError("incomplete threshold scoring")
    value, maxima = rollout_threshold(rows, ROLE_INDICES["threshold"])
    manifest = dict(
        status="PASS", scope="sorting_development_smoke_only", device=str(device), gpu=gpu_info, seed=seed,
        epochs=2, batch_size=32, learning_rate=1e-3, optimizer="Adam", gradient_clip=1.0,
        model_config=config, history_length=8, epoch_losses=history,
        selected_epoch=best_epoch, selected_loss=best_loss, checkpoint=str(checkpoint.resolve()),
        checkpoint_reload_equal=True, normalization_source="representation",
        normalization_obs_rows=normalizer.state_dict()["obs_count"],
        normalization_action_rows=normalizer.state_dict()["action_count"],
        roles={k: list(v) for k, v in ROLE_INDICES.items()},
        sample_counts={"representation": len(train.dataset), "selection": len(selection.dataset),
                       "threshold": len(threshold.dataset)},
        threshold=dict(style="ct_quantile", quantile=0.9, window=1,
                       aggregation="per_rollout_max", interpolation="linear", value=value,
                       rollout_maxima=maxima, finite_sample_guarantee_claimed=False),
        score="softplus(-logit)", provenance=provenance,
        performance_evaluated=False,
    )
    (output / "threshold_scores.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"selected_epoch={best_epoch}, checkpoint_reload=OK, threshold={value:.6f}")
    print(f"{'GPU' if gpu_uuid else 'CPU'} TRAINING PASS: {output.resolve()}")
    return manifest
