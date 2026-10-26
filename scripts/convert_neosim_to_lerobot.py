#!/usr/bin/env python3
r"""Convert NeoSim dual-arm HDF5 episodes into a LeRobot dataset for ``sim_dual_arm_tactile``.

Example:
  python scripts/convert_neosim_to_lerobot.py \
    --src /path/to/neosim/data/dual_cup_handover_place/collect_vtla/hdf5 \
    --dst /path/to/datasets/neosim_vtla/dual_cup_handover_place \
    --prompt "Use both arms to hand over the cup and place it" \
    --fps 60

``--src`` is one task's ``hdf5/`` directory as NeoSim writes it
(``data/<task>/<config>/hdf5/<seed>.hdf5``); the ``<config>`` directory that contains ``hdf5/``
is accepted too. Episodes are converted in ascending numeric seed order.

An episode with T saved frames yields T - 1 dataset frames, t = 0 .. T - 2:

  observation.state[t] = concat(embodiment/joint[t, :8],     embodiment_b/joint[t, :8])
  action[t]            = concat(embodiment/joint[t + 1, :8], embodiment_b/joint[t + 1, :8])

Each arm contributes its 7 joints plus the first finger joint as the gripper, the same
definition as NeoSim's ``policy/ACT/process_data_dual.py``. Both are stored as absolute float32
values: ``LeRobotSimTactileJointDataConfig`` turns the joints into deltas and keeps the grippers
absolute. The last frame is dropped because it has no next joint target. Frame 0 is always kept:
the data config loads it as the tactile baseline (tac_0) of every frame in the episode.

Images are stored as videos at their native resolution, without color conversion:

  observation.images.top        <- observation/head/rgb           (270x480 in NeoSim)
  observation.images.wrist_l    <- observation/wrist/rgb
  observation.images.wrist_r    <- observation/wrist_b/rgb
  observation.images.tactile_a  <- tactile/left_tactile/rgb       (240x320 in NeoSim)
  observation.images.tactile_b  <- tactile/right_tactile/rgb
  observation.images.tactile_c  <- tactile/left_tactile_b/rgb
  observation.images.tactile_d  <- tactile/right_tactile_b/rgb

NeoSim stores every frame as JPEG bytes produced by ``cv2.imencode`` on the simulator's RGB
array, with no color conversion (``envs/utils/data.py``, ``HDF5Handler.img_to_stream``), so
``cv2.imdecode`` returns the simulator's own channel order and the frames go into the video as
RGB unchanged. NeoSim's own decode helper is not used: with ``resize=True`` it squashes every
frame to 256x256, whereas the model transform letterboxes the native frame.

Tactile videos use x264 lossless coding by default (``--tactile-crf 0``; only the yuv420p chroma
subsampling remains), camera videos use ``--crf``. The model reads the tactile difference
tac_t - tac_0 and online evaluation sends lossless PNG frames, so lossy coding would add a noise floor
to that difference that evaluation never sees: at crf 23, bit-identical free-space frames decode into
dozens of different images.

The output uses the per-episode LeRobot v2.1 layout of ``scripts/convert_canonical_data.py``,
which ``n0vtla/training/data_loader.py`` reads directly:

  data/chunk-000/episode_000000.parquet
  videos/chunk-000/<video_key>/episode_000000.mp4    (h264, yuv420p)
  meta/info.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  meta/episodes_stats.jsonl
  meta/source_episode_map.json                        (output episode -> NeoSim seed)
  metadata.json
  conversion_report.json

``conversion_report.json`` is written first and names this converter; ``--overwrite`` only replaces a
directory that carries that marker, so it can redo an interrupted run but never deletes another dataset
(``convert_canonical_data.py`` writes the same top-level entries). ``meta/info.json`` is written last,
so an interrupted run never looks like a complete dataset.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import concurrent.futures
import contextlib
import fractions
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import av
import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_FPS = 60  # NeoSim saves one frame every 2 simulation steps of 1/120 s.
CHUNK_SIZE = 1000
ARM_JOINT_DIM = 8  # Per arm: 7 joints + the first finger joint, used as the gripper.
RAW_ACTION_DIM = 2 * ARM_JOINT_DIM
ROBOT_TYPE = "neosim_dual_arm_tactile"
CONVERTER_NAME = "convert_neosim_to_lerobot"  # Marker in conversion_report.json; see _foreign_output_reason.

VIDEO_ENCODER = "libx264"
VIDEO_CODEC = "h264"
VIDEO_PIX_FMT = "yuv420p"

# Arm A, arm B; NeoSim stores each as (T, 9): 7 joints + 2 finger joints.
JOINT_KEYS = ("embodiment/joint", "embodiment_b/joint")

# LeRobot video key -> NeoSim HDF5 dataset holding one JPEG per saved frame.
IMAGE_KEY_MAP = {
    "observation.images.top": "observation/head/rgb",
    "observation.images.wrist_l": "observation/wrist/rgb",
    "observation.images.wrist_r": "observation/wrist_b/rgb",
    "observation.images.tactile_a": "tactile/left_tactile/rgb",
    "observation.images.tactile_b": "tactile/right_tactile/rgb",
    "observation.images.tactile_c": "tactile/left_tactile_b/rgb",
    "observation.images.tactile_d": "tactile/right_tactile_b/rgb",
}
# Encoded with --tactile-crf (lossless by default) instead of --crf.
TACTILE_VIDEO_KEYS = frozenset(key for key, source in IMAGE_KEY_MAP.items() if source.startswith("tactile/"))

# Everything this script writes into the output root. The same names alone do not prove that a
# directory is ours (convert_canonical_data.py writes them too), so --overwrite also needs the marker.
OUTPUT_ENTRIES = frozenset({"data", "videos", "meta", "metadata.json", "conversion_report.json"})


class _EpisodeError(ValueError):
    """An episode that cannot be converted. It is reported and skipped; the run continues."""


def _collect_episode_files(src: Path) -> tuple[Path, list[tuple[int, Path]]]:
    hdf5_dir = src / "hdf5" if (src / "hdf5").is_dir() else src
    if not hdf5_dir.is_dir():
        raise SystemExit(f"source directory not found: {src}")
    paths = sorted(hdf5_dir.glob("*.hdf5"))
    non_seed = [path.name for path in paths if not path.stem.isdigit()]
    if non_seed:
        raise SystemExit(f"expected NeoSim <seed>.hdf5 files in {hdf5_dir}, found {non_seed[:5]}")
    return hdf5_dir, sorted((int(path.stem), path) for path in paths)


def _read_json_object(path: Path) -> dict[str, Any]:
    """The JSON object stored at ``path``, or {} if it is missing, unreadable or not an object."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _foreign_output_reason(path: Path, entries: list[str]) -> str | None:
    """Why the non-empty directory ``path`` is not this converter's output, or None if it is (complete
    or from an interrupted run)."""
    foreign = [name for name in entries if name not in OUTPUT_ENTRIES]
    if foreign:
        return f"it holds entries this converter does not write: {foreign[:5]}"
    if _read_json_object(path / "conversion_report.json").get("converter") != CONVERTER_NAME:
        return f"its conversion_report.json does not name {CONVERTER_NAME}, so it may be another LeRobot dataset"
    info_path = path / "meta" / "info.json"
    if info_path.exists() and _read_json_object(info_path).get("robot_type") != ROBOT_TYPE:
        return f"its meta/info.json robot_type is not {ROBOT_TYPE}"
    return None


def _reset_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        entries = sorted(entry.name for entry in path.iterdir())
        if entries and not overwrite:
            raise FileExistsError(f"output directory is not empty: {path}; pass --overwrite to replace it")
        reason = _foreign_output_reason(path, entries) if entries else None
        if reason:
            raise FileExistsError(f"refusing to replace {path}: {reason}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def _open_episode(path: Path) -> Iterator[h5py.File]:
    """Open one NeoSim episode. An HDF5 error (truncated or empty copy, corrupt dataset) drops the
    episode like any other bad episode instead of aborting the whole run."""
    try:
        with h5py.File(path, "r") as f:
            yield f
    except _EpisodeError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise _EpisodeError(f"unreadable_hdf5: {error}") from error


def _decode_frame(buf: bytes) -> np.ndarray | None:
    """Decode one stored JPEG. No BGR/RGB flip: NeoSim encoded the simulator array with cv2,
    so cv2 decoding returns the simulator's original (RGB) channel order."""
    data = np.frombuffer(buf, dtype=np.uint8)
    if data.size == 0:  # cv2.imdecode raises on an empty buffer instead of returning None.
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _read_joints(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    joints = []
    for key in JOINT_KEYS:
        joint = np.asarray(f[key][()])
        if joint.ndim != 2 or joint.shape[1] < ARM_JOINT_DIM:
            raise _EpisodeError(f"bad_shape_{key}={joint.shape}")
        if not np.isfinite(joint).all():
            raise _EpisodeError(f"non_finite_{key}")
        joints.append(joint)
    if joints[0].shape[0] != joints[1].shape[0]:
        raise _EpisodeError(f"arm_frame_count_mismatch={joints[0].shape[0]}_vs_{joints[1].shape[0]}")
    return joints[0], joints[1]


def _check_frame_count(frames: h5py.Dataset, source_key: str, num_source_frames: int) -> None:
    if frames.ndim != 1 or frames.dtype.kind not in ("S", "O"):
        raise _EpisodeError(f"not_encoded_frames_{source_key}=dtype_{frames.dtype}_shape_{frames.shape}")
    if frames.shape[0] != num_source_frames:
        raise _EpisodeError(f"frame_count_mismatch_{source_key}={frames.shape[0]}_expected_{num_source_frames}")


def _inspect_episode(path: Path) -> dict[str, Any]:
    """Check keys, frame counts and image sizes of one episode, decoding only its first frames."""
    with _open_episode(path) as f:
        source_keys = (*JOINT_KEYS, *IMAGE_KEY_MAP.values())
        missing = [key for key in source_keys if key not in f]
        if missing:
            raise _EpisodeError(f"missing_keys={missing}")
        not_datasets = [key for key in source_keys if not isinstance(f[key], h5py.Dataset)]
        if not_datasets:
            raise _EpisodeError(f"not_datasets={not_datasets}")
        joint_a, _ = _read_joints(f)
        num_source_frames = joint_a.shape[0]
        if num_source_frames < 2:
            raise _EpisodeError(f"too_short={num_source_frames}_frames")
        image_shapes: dict[str, tuple[int, ...]] = {}
        for video_key, source_key in IMAGE_KEY_MAP.items():
            frames = f[source_key]
            _check_frame_count(frames, source_key, num_source_frames)
            image = _decode_frame(frames[0])
            if image is None:
                raise _EpisodeError(f"undecodable_frame_{source_key}[0]")
            image_shapes[video_key] = image.shape
    return {"num_source_frames": num_source_frames, "image_shapes": image_shapes}


def _encode_video(
    frames: np.ndarray,
    num_frames: int,
    path: Path,
    *,
    shape: tuple[int, ...],
    source_key: str,
    fps: int,
    crf: int,
    gop: int,
) -> int:
    """Decode the first ``num_frames`` stored JPEGs and encode them as one mp4 at frame times t / fps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = shape[:2]
    time_base = fractions.Fraction(1, fps)
    with av.open(str(path), mode="w") as container:
        # No B-frames: decode order equals presentation order, which keeps timestamp seeks exact.
        stream = container.add_stream(VIDEO_ENCODER, rate=fps, options={"crf": str(crf), "g": str(gop), "bf": "0"})
        stream.width = width
        stream.height = height
        stream.pix_fmt = VIDEO_PIX_FMT
        for t in range(num_frames):
            image = _decode_frame(frames[t])
            if image is None:
                raise _EpisodeError(f"undecodable_frame_{source_key}[{t}]")
            if image.shape != shape:
                raise _EpisodeError(f"frame_shape_{source_key}[{t}]={image.shape}_expected_{shape}")
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = t
            frame.time_base = time_base
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    with av.open(str(path)) as container:
        written = container.streams.video[0].frames
    if written != num_frames:
        raise RuntimeError(f"{path} holds {written} frames, expected {num_frames}")
    return path.stat().st_size


def _feature_stats(values: np.ndarray) -> dict[str, list[float | int]]:
    """Per-episode LeRobot v2.1 stats of one numeric column."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def _fixed_size_list_array(values: np.ndarray, width: int, value_type: pa.DataType) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=value_type), width)


def _hf_schema_metadata() -> dict[bytes, bytes]:
    vector = {"feature": {"dtype": "float32", "_type": "Value"}, "length": RAW_ACTION_DIM, "_type": "Sequence"}
    features = {
        "observation.state": vector,
        "action": vector,
        "timestamp": {"dtype": "float32", "_type": "Value"},
        "frame_index": {"dtype": "int64", "_type": "Value"},
        "episode_index": {"dtype": "int64", "_type": "Value"},
        "index": {"dtype": "int64", "_type": "Value"},
        "task_index": {"dtype": "int64", "_type": "Value"},
    }
    return {b"huggingface": json.dumps({"info": {"features": features}}).encode("utf-8")}


def _write_episode_parquet(
    path: Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    timestamp: np.ndarray,
    frame_index: np.ndarray,
    episode_index: int,
    index: np.ndarray,
) -> int:
    length = state.shape[0]
    table = pa.Table.from_arrays(
        [
            _fixed_size_list_array(state, RAW_ACTION_DIM, pa.float32()),
            _fixed_size_list_array(action, RAW_ACTION_DIM, pa.float32()),
            pa.array(timestamp, type=pa.float32()),
            pa.array(frame_index, type=pa.int64()),
            pa.array(np.full(length, episode_index, dtype=np.int64), type=pa.int64()),
            pa.array(index, type=pa.int64()),
            pa.array(np.zeros(length, dtype=np.int64), type=pa.int64()),
        ],
        names=["observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"],
    ).replace_schema_metadata(_hf_schema_metadata())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path.stat().st_size


def _video_feature(shape: tuple[int, ...], fps: int) -> dict[str, Any]:
    height, width, channels = shape
    return {
        "dtype": "video",
        "shape": [height, width, channels],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": VIDEO_CODEC,
            "video.pix_fmt": VIDEO_PIX_FMT,
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": channels,
            "has_audio": False,
        },
    }


def _build_info_json(
    *,
    total_episodes: int,
    total_frames: int,
    fps: int,
    data_size_bytes: int,
    video_size_bytes: int,
    image_shapes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [RAW_ACTION_DIM], "names": None},
        "action": {"dtype": "float32", "shape": [RAW_ACTION_DIM], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for video_key, shape in image_shapes.items():
        features[video_key] = _video_feature(shape, fps)

    return {
        "codebase_version": "v2.1",
        "robot_type": ROBOT_TYPE,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * len(image_shapes),
        "total_chunks": math.ceil(total_episodes / CHUNK_SIZE),
        "chunks_size": CHUNK_SIZE,
        "data_files_size_in_mb": math.ceil(data_size_bytes / (1024 * 1024)),
        "video_files_size_in_mb": math.ceil(video_size_bytes / (1024 * 1024)),
        "fps": float(fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _convert_episode(
    path: Path,
    *,
    episode_index: int,
    output_dir: Path,
    image_shapes: dict[str, tuple[int, ...]],
    executor: concurrent.futures.ThreadPoolExecutor,
    fps: int,
    crf: int,
    tactile_crf: int,
    gop: int,
) -> dict[str, Any]:
    """Write one episode's videos (in parallel) and parquet. Returns its arrays and sizes."""
    with _open_episode(path) as f:
        joint_a, joint_b = _read_joints(f)
        num_source_frames = joint_a.shape[0]
        encoded = {}
        for video_key, source_key in IMAGE_KEY_MAP.items():
            frames = f[source_key]
            _check_frame_count(frames, source_key, num_source_frames)
            encoded[video_key] = frames[()]
    length = num_source_frames - 1

    episode_chunk = episode_index // CHUNK_SIZE
    video_paths = {
        video_key: output_dir / "videos" / f"chunk-{episode_chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"
        for video_key in IMAGE_KEY_MAP
    }
    futures = {
        video_key: executor.submit(
            _encode_video,
            encoded[video_key],
            length,
            video_paths[video_key],
            shape=image_shapes[video_key],
            source_key=IMAGE_KEY_MAP[video_key],
            fps=fps,
            crf=tactile_crf if video_key in TACTILE_VIDEO_KEYS else crf,
            gop=gop,
        )
        for video_key in IMAGE_KEY_MAP
    }
    concurrent.futures.wait(futures.values())
    errors = [future.exception() for future in futures.values() if future.exception() is not None]
    if errors:
        for video_path in video_paths.values():
            video_path.unlink(missing_ok=True)
        for error in errors:
            if not isinstance(error, _EpisodeError):
                raise error
        raise _EpisodeError("; ".join(str(error) for error in errors))
    video_size_bytes = sum(future.result() for future in futures.values())

    # state[t] = joint[t], action[t] = joint[t + 1]; frame 0 stays (it is the tactile baseline).
    state = np.concatenate([joint_a[:-1, :ARM_JOINT_DIM], joint_b[:-1, :ARM_JOINT_DIM]], axis=1).astype(np.float32)
    action = np.concatenate([joint_a[1:, :ARM_JOINT_DIM], joint_b[1:, :ARM_JOINT_DIM]], axis=1).astype(np.float32)
    return {
        "length": length,
        "num_source_frames": num_source_frames,
        "state": state,
        "action": action,
        "video_size_bytes": video_size_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="NeoSim hdf5/ directory of one task, or its parent.")
    parser.add_argument("--dst", type=Path, required=True, help="Output local LeRobot dataset root.")
    parser.add_argument(
        "--prompt",
        required=True,
        help="Task string written to meta/tasks.jsonl. Evaluation must send exactly the same prompt.",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Dataset fps (NeoSim records at 60 Hz).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an earlier output of this converter; any other non-empty directory is refused.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert at most this many valid episodes, taken in ascending seed order.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="libx264 constant rate factor of the camera videos; lower is higher quality.",
    )
    parser.add_argument(
        "--tactile-crf",
        type=int,
        default=0,
        help="libx264 constant rate factor of the tactile videos. 0 is lossless: lossy coding adds noise to "
        "tac_t - tac_0 that the lossless online frames do not have.",
    )
    parser.add_argument(
        "--gop",
        type=int,
        default=2,
        help="Keyframe interval. Training decodes random frames, so short GOPs keep seeking cheap.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=len(IMAGE_KEY_MAP),
        help="Threads that decode and encode an episode's videos in parallel.",
    )
    args = parser.parse_args()

    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.max_episodes is not None and args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    if not (0 <= args.crf <= 51 and 0 <= args.tactile_crf <= 51):
        parser.error("--crf and --tactile-crf must be in [0, 51]")
    if args.gop <= 0 or args.num_workers <= 0:
        parser.error("--gop and --num-workers must be positive")

    hdf5_dir, episode_files = _collect_episode_files(args.src)
    if not episode_files:
        raise SystemExit(f"no <seed>.hdf5 episodes found under {hdf5_dir}")

    # Validate before touching the output: keys, frame counts and one image size per camera. Every
    # file is inspected even with --max-episodes, because an episode can still be dropped while it is
    # converted; the next valid one then takes its place.
    dropped: list[dict[str, Any]] = []
    candidates: list[tuple[int, Path]] = []
    image_shapes: dict[str, tuple[int, ...]] = {}
    for seed, path in episode_files:
        try:
            inspected = _inspect_episode(path)
        except _EpisodeError as error:
            dropped.append({"source_episode": path.name, "seed": seed, "reasons": [str(error)]})
            continue
        for video_key, shape in inspected["image_shapes"].items():
            if video_key not in image_shapes:
                image_shapes[video_key] = shape
            elif shape != image_shapes[video_key]:
                raise ValueError(
                    f"inconsistent image size for {IMAGE_KEY_MAP[video_key]}: {path} has {shape}, earlier "
                    f"episodes have {image_shapes[video_key]}; convert different resolutions into separate datasets"
                )
        candidates.append((seed, path))
    if not candidates:
        raise SystemExit(f"no convertible episodes under {hdf5_dir}; dropped: {dropped}")
    odd = {key: shape for key, shape in image_shapes.items() if shape[0] % 2 or shape[1] % 2}
    if odd:
        raise ValueError(f"{VIDEO_PIX_FMT} video needs even image sizes, got {odd}")

    _reset_output_dir(args.dst, overwrite=args.overwrite)
    # Mark the directory as ours before writing anything else; the full report replaces this at the end.
    _write_json(args.dst / "conversion_report.json", {"converter": CONVERTER_NAME, "complete": False})
    (args.dst / "meta").mkdir(parents=True, exist_ok=True)

    kept: list[dict[str, Any]] = []
    episodes_rows: list[dict[str, Any]] = []
    episodes_stats_rows: list[dict[str, Any]] = []
    current_global_index = 0
    data_size_bytes = 0
    video_size_bytes = 0
    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for position, (seed, path) in enumerate(candidates, start=1):
            if args.max_episodes is not None and len(kept) == args.max_episodes:
                break
            episode_index = len(kept)
            episode_start = time.perf_counter()
            try:
                episode = _convert_episode(
                    path,
                    episode_index=episode_index,
                    output_dir=args.dst,
                    image_shapes=image_shapes,
                    executor=executor,
                    fps=args.fps,
                    crf=args.crf,
                    tactile_crf=args.tactile_crf,
                    gop=args.gop,
                )
            except _EpisodeError as error:
                dropped.append({"source_episode": path.name, "seed": seed, "reasons": [str(error)]})
                print(f"[{position}/{len(candidates)}] seed {seed}: dropped ({error})", flush=True)
                continue

            length = episode["length"]
            frame_index = np.arange(length, dtype=np.int64)
            timestamp = (frame_index.astype(np.float32) / np.float32(args.fps)).astype(np.float32, copy=False)
            dataset_index = np.arange(current_global_index, current_global_index + length, dtype=np.int64)
            parquet_path = (
                args.dst / "data" / f"chunk-{episode_index // CHUNK_SIZE:03d}" / f"episode_{episode_index:06d}.parquet"
            )
            data_size_bytes += _write_episode_parquet(
                parquet_path,
                state=episode["state"],
                action=episode["action"],
                timestamp=timestamp,
                frame_index=frame_index,
                episode_index=episode_index,
                index=dataset_index,
            )
            video_size_bytes += episode["video_size_bytes"]

            episodes_rows.append({"episode_index": episode_index, "tasks": [args.prompt], "length": length})
            episodes_stats_rows.append(
                {
                    "episode_index": episode_index,
                    "stats": {
                        "observation.state": _feature_stats(episode["state"]),
                        "action": _feature_stats(episode["action"]),
                        "timestamp": _feature_stats(timestamp),
                        "frame_index": _feature_stats(frame_index),
                        "episode_index": _feature_stats(np.full(length, episode_index, dtype=np.int64)),
                        "index": _feature_stats(dataset_index),
                        "task_index": _feature_stats(np.zeros(length, dtype=np.int64)),
                    },
                }
            )
            kept.append(
                {
                    "output_episode_index": episode_index,
                    "source_episode": path.name,
                    "seed": seed,
                    "source_frames": episode["num_source_frames"],
                    "length": length,
                }
            )
            current_global_index += length
            print(
                f"[{position}/{len(candidates)}] seed {seed} -> episode {episode_index}: "
                f"{length} frames ({time.perf_counter() - episode_start:.1f}s)",
                flush=True,
            )

    if not kept:
        raise SystemExit(f"no episodes left after conversion; dropped: {dropped}")
    # Valid episodes left out by --max-episodes.
    converted = {row["source_episode"] for row in (*kept, *dropped)}
    unused = [{"source_episode": path.name, "seed": seed} for seed, path in candidates if path.name not in converted]

    total_frames = current_global_index
    lengths = [row["length"] for row in kept]
    _write_jsonl(args.dst / "meta" / "tasks.jsonl", [{"task_index": 0, "task": args.prompt}])
    _write_jsonl(args.dst / "meta" / "episodes.jsonl", episodes_rows)
    _write_jsonl(args.dst / "meta" / "episodes_stats.jsonl", episodes_stats_rows)
    _write_json(args.dst / "meta" / "source_episode_map.json", kept)
    _write_json(
        args.dst / "metadata.json",
        {
            "metadata_version": 1,
            "robot_type": ROBOT_TYPE,
            "fps": float(args.fps),
            "canonical_schema": False,
            "state_source_key": "concat(embodiment/joint[t, :8], embodiment_b/joint[t, :8])",
            "action_source_key": "concat(embodiment/joint[t + 1, :8], embodiment_b/joint[t + 1, :8])",
            "raw_action_dim": RAW_ACTION_DIM,
            "total_episodes": len(kept),
            "total_frames": total_frames,
        },
    )
    _write_json(
        args.dst / "conversion_report.json",
        {
            "converter": CONVERTER_NAME,
            "complete": True,
            "source_dir": str(hdf5_dir.resolve()),
            "task": args.prompt,
            "fps": args.fps,
            "image_sources": IMAGE_KEY_MAP,
            "video": {
                "codec": VIDEO_CODEC,
                "pix_fmt": VIDEO_PIX_FMT,
                "crf": args.crf,
                "tactile_crf": args.tactile_crf,
                "gop": args.gop,
            },
            "max_episodes": args.max_episodes,
            "kept_episode_count": len(kept),
            "dropped_episode_count": len(dropped),
            "unused_episode_count": len(unused),
            "kept_episodes": kept,
            "dropped_episodes": dropped,
            "unused_episodes": unused,
        },
    )
    _write_json(
        args.dst / "meta" / "info.json",
        _build_info_json(
            total_episodes=len(kept),
            total_frames=total_frames,
            fps=args.fps,
            data_size_bytes=data_size_bytes,
            video_size_bytes=video_size_bytes,
            image_shapes=image_shapes,
        ),
    )

    print(f"output_dir: {args.dst}")
    print(f"source_dir: {hdf5_dir}")
    print(f"task: {args.prompt}")
    print(f"kept episodes: {len(kept)} (seeds {kept[0]['seed']}..{kept[-1]['seed']})")
    if args.max_episodes is not None and len(kept) < args.max_episodes:
        print(f"WARNING: --max-episodes {args.max_episodes} requested, but only {len(kept)} valid episodes exist")
    print(f"dropped episodes: {len(dropped)}")
    for row in dropped:
        print(f"  {row['source_episode']}: {'; '.join(row['reasons'])}")
    if unused:
        print(f"unused valid episodes (beyond --max-episodes): {len(unused)} (seeds {[row['seed'] for row in unused]})")
    print(
        f"total frames: {total_frames} "
        f"(per episode min/mean/max {min(lengths)}/{np.mean(lengths):.1f}/{max(lengths)})"
    )
    print(f"fps: {args.fps}")
    print(f"state width: {RAW_ACTION_DIM}")
    print(f"action width: {RAW_ACTION_DIM}")
    for video_key, shape in image_shapes.items():
        print(f"{video_key}: {tuple(shape)} <- {IMAGE_KEY_MAP[video_key]}")
    print(
        f"videos: {VIDEO_CODEC} {VIDEO_PIX_FMT} crf={args.crf} tactile_crf={args.tactile_crf} gop={args.gop}, "
        f"{video_size_bytes / (1024 * 1024):.1f} MB; parquet {data_size_bytes / 1024:.1f} KB"
    )
    print(f"elapsed: {time.perf_counter() - start_time:.1f}s")


if __name__ == "__main__":
    main()
