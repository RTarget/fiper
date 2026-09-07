from pathlib import Path
import pickle
import numpy as np
import torch


TASKS = [
    "sorting",
    "stacking",
    "push_t",
    "pretzel",
    "push_chair",
]

TENSOR_NAMES = [
    "obs_embeddings",
    "action_preds",
    "states",
    "rgb_images",
]


def load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def describe_tensor(path: Path):
    if not path.exists():
        return f"{path.stem}: missing"

    try:
        tensor = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        shape = tuple(tensor.shape) if hasattr(tensor, "shape") else None
        dtype = getattr(tensor, "dtype", None)

        if isinstance(tensor, torch.Tensor):
            finite = bool(torch.isfinite(tensor).all().item())
            return (
                f"{path.stem}: shape={shape}, "
                f"dtype={dtype}, finite={finite}"
            )

        return f"{path.stem}: type={type(tensor).__name__}"

    except Exception as exc:
        return f"{path.stem}: ERROR {type(exc).__name__}: {exc}"


def describe_labels(metadata: dict, key: str):
    if key not in metadata:
        return f"{key}: missing"

    values = np.asarray(metadata[key])

    if values.dtype == bool:
        return (
            f"{key}: length={len(values)}, "
            f"true={int(values.sum())}, "
            f"false={int((~values).sum())}"
        )

    return (
        f"{key}: length={len(values)}, "
        f"first={values[:5].tolist()}, "
        f"last={values[-5:].tolist()}"
    )


def inspect_task(task: str):
    print(f"\n===== {task} =====")

    processed_dir = Path("data") / task / "processed_rollouts"
    metadata_path = processed_dir / "metadata.pkl"

    if not processed_dir.exists():
        print(f"processed_rollouts missing: {processed_dir}")
        return

    if not metadata_path.exists():
        print("metadata.pkl missing")
        return

    metadata = load_pickle(metadata_path)

    print(f"processed_dir: {processed_dir}")
    print(f"metadata keys: {sorted(metadata.keys())}")

    scalar_keys = [
        "num_steps",
        "num_rollouts",
        "num_robots",
    ]

    for key in scalar_keys:
        print(f"{key}: {metadata.get(key, 'missing')}")

    label_keys = [
        "successful_rollout_labels",
        "failed_rollout_labels",
        "calibration_rollout_labels",
        "test_rollout_labels",
        "id_rollout_labels",
        "ood_rollout_labels",
    ]

    for key in label_keys:
        print(describe_labels(metadata, key))

    episode_lengths = metadata.get("episode_lengths")
    if episode_lengths is not None:
        lengths = np.asarray(episode_lengths)
        print(
            "episode_lengths: "
            f"count={len(lengths)}, "
            f"min={int(lengths.min())}, "
            f"max={int(lengths.max())}, "
            f"mean={float(lengths.mean()):.2f}"
        )

    print("tensors:")
    for tensor_name in TENSOR_NAMES:
        tensor_path = processed_dir / f"{tensor_name}.pt"
        print(f"  {describe_tensor(tensor_path)}")


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    for index, task in enumerate(TASKS):
        inspect_task(task)

    print("\nAudit finished.")


if __name__ == "__main__":
    main()