#!/usr/bin/env python
"""Contact statistics for 1A input resolution: per-view thresholds tau_k, floor tau_min, P(n_a).

WHAT THIS COMPUTES
------------------
For every frame t of every episode e and every tactile view k of a local LeRobot dataset, the
contact strength the 1A allocator uses (n0vtla/models_pytorch/input_resolution.py):

    s_k(e, t) = contact_strength(tac_t, tac_0)          tac_0 = the episode's frame 0

with EXACTLY the model's preprocessing, reusing the repo's own code for every step: the training
video decoder (LocalLeRobotV3Dataset, torchcodec when importable, else precise pyav), then
FlexivTactileInputs' ``_parse_image`` (float -> uint8 HWC), ``ResizeImages(224, 224)`` (the
``resize_with_pad`` letterbox) and Observation.from_dict's uint8 -> [-1, 1] conversion. Then, per view:

    mu_k, sigma_k  mean / std of s over the first N0 frames after the baseline, t = 1..N0, of every
                   episode (the no-contact noise level: the model's input is "minus frame 0", so the
                   start is assumed contact-free; frame 0 itself is the baseline, s = 0 exactly, and
                   would only bias the estimate), excluding episodes whose start is already
                   elevated: mean_{1<=t<=N0} s > median + 5 MAD over episodes
    tau_floor      --tau-floor otsu: Otsu threshold of log s over all frames after the baselines, all
                   views (rendered sim frames are nearly noise-free, so sigma can be ~0 and the floor
                   must sit above the video-compression noise); or a fixed value (default 0: no floor).
                   Frames with s == 0 exactly (bit-identical to the baseline: free space under lossless
                   tactile video) are the no-contact class there; they enter the histogram at the
                   resolution of s (one pixel, one channel, one uint8 level: (2/255) / (3 pool^2 k)),
                   and a split through empty bins is taken in their middle, so a clean gap between
                   those zeros and the first contacts is cut in the middle of the gap (log scale)
    tau_k          max(mu_k + 3 sigma_k, tau_floor)
    P(n_a)         n_a(e, t) = number of views with s_k > tau_k, over all frames

WHAT IT WRITES
--------------
--out (JSON, read by the model through VTLA_1A_STATS / N0VTLAConfig.input_res_stats):
    tau, tau_floor, mu, sigma, na_hist, content_box, topq, pool, n0, views, per-view statistics
    (with zero_frac, the share of frames after the baseline with s == 0 exactly), excluded
    episodes, the Otsu diagnostics and the rule allocation this produces on the data.
--plot-dir (SVG; the vtla env has no matplotlib):
    hist_log_s.svg     per-view histogram of log10 s with mu + 3 sigma, tau_floor and tau; the s == 0
                       frames are the separate bar at "0"
    episode_<e>.svg    s_k(t) with tau_k for a few episodes, plus the resulting rule allocation n_k(t)

CHECK (1A technical plan 8.2, sim plan S6): the log s histograms should be bimodal with tau in the
valley; s_k(t) should cross tau_k exactly when the video shows contact; P(n_a >= 1) is the share of
frames where 1A changes anything. When most no-contact frames are exactly 0 (logged as a warning),
tau_min sits in the gap above them and any gel change counts as contact: if the histogram shows a
low mode of positive s that is NOT contact (gel jitter in free space), pass --tau-floor <value> above it.

RUN
---
    export PYTHONPATH=$PWD
    python scripts/tactile_contact_stats.py \\
        --repo-id /mnt/ssd2/guningquan/Dataset/neosim_vtla/dual_cup_handover_place \\
        --content-box 28 196 0 224 --n0 10 --top-q 0.05 --pool 4 --tau-floor otsu \\
        --out assets/sim_dual_1a_rule/neosim_cup_handover/tactile_contact_stats.json \\
        --plot-dir logs/contact_stats_sim_cup_handover --num-workers 8

The content box defaults to the letterbox geometry of the dataset's tactile frames; content box,
top-q and pool must match the model config (input_res_*), which the model checks when it loads the JSON.
"""
from __future__ import annotations

import os
import sys

# Letterboxing runs through JAX (resize_with_pad, exactly as in training). Keep JAX on the CPU so it
# never preallocates GPU memory.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# --- make `import n0vtla` resolve to THIS checkout when run as `python scripts/tactile_contact_stats.py`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse  # noqa: E402
import html  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import multiprocessing  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from n0vtla import transforms as _transforms  # noqa: E402
from n0vtla.models import model as _model  # noqa: E402
from n0vtla.models_pytorch import input_resolution as _ir  # noqa: E402
from n0vtla.policies.flexiv_policy import _parse_image  # noqa: E402
from n0vtla.training import data_loader as _data  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True, help="local LeRobot dataset root (v2.1 or v3)")
    p.add_argument("--tactile-keys", nargs="+", default=None,
                   help="tactile video columns in MODEL view order (default: every video column whose name "
                        "contains 'tactile', sorted; that is tactile_a..d / the canonical TACTILE_SHORT_KEYS order)")
    p.add_argument("--content-box", nargs=4, type=int, default=None, metavar=("R0", "R1", "C0", "C1"),
                   help="content region of the 224x224 letterboxed frame (default: the letterbox geometry of "
                        "the dataset's tactile frames, e.g. 28 196 0 224 for 240x320, 0 224 48 176 for 700x400)")
    p.add_argument("--n0", type=int, default=10,
                   help="frames after the baseline (t = 1..N0) treated as contact-free noise samples")
    p.add_argument("--top-q", type=float, default=0.05, help="fraction of pooled cells averaged into s")
    p.add_argument("--pool", type=int, default=4, help="average-pooling cell size before the top-q")
    p.add_argument("--n-sigma", type=float, default=3.0, help="tau_k = mu_k + n_sigma * sigma_k (before the floor)")
    p.add_argument("--start-mad", type=float, default=5.0,
                   help="exclude an episode from a view's mu/sigma if its start s > median + start_mad * MAD")
    p.add_argument("--tau-floor", default="0",
                   help="'otsu' (Otsu threshold of log s over all frames) or a fixed value; tau_k = max(.., floor)")
    p.add_argument("--sibling", nargs="+", type=int, default=list(_ir.DEFAULT_SIBLING),
                   help="other finger of the same gripper per view (only for the allocation diagnostics)")
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--plot-dir", default=None, help="directory for the SVG diagnostics (skipped if unset)")
    p.add_argument("--plot-episodes", type=int, default=4, help="number of episodes drawn as s_k(t) plots")
    p.add_argument("--max-episodes", type=int, default=None, help="only use the first N episodes")
    p.add_argument("--num-workers", type=int, default=0, help="decoding processes (0 = in-process)")
    p.add_argument("--chunk", type=int, default=128, help="frames decoded per video read")
    p.add_argument("--tolerance-s", type=float, default=0.04, help="video timestamp tolerance (as the sim configs)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model preprocessing (the training transforms, reused)
# ---------------------------------------------------------------------------
_RESIZE = _transforms.ResizeImages(*_model.IMAGE_RESOLUTION)


def model_tactile_frames(frames: torch.Tensor) -> torch.Tensor:
    """Decoded frames (N, 3, H, W) float in [0, 1] -> the model's tactile tensors (N, 3, 224, 224) in [-1, 1].

    Same chain as a training sample: FlexivTactileInputs' ``_parse_image`` (float -> uint8 HWC), the
    model transform ``ResizeImages(224, 224)`` (resize_with_pad letterbox, applied per image as in
    the data pipeline), then Observation.from_dict's torch branch (uint8 -> [-1, 1], NCHW).
    """
    imgs = [np.asarray(_RESIZE({"image": {"tac": _parse_image(f.numpy())}})["image"]["tac"]) for f in frames]
    x = torch.from_numpy(np.stack(imgs))
    return x.to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0


def letterbox_content_box(height: int, width: int) -> tuple[int, int, int, int]:
    """(r0, r1, c0, c1) of the image area after ``ResizeImages`` letterboxes an (height, width) frame."""
    probe = np.full((height, width, 3), 255, dtype=np.uint8)
    out = np.asarray(_RESIZE({"image": {"tac": probe}})["image"]["tac"])[..., 0]
    rows, cols = np.nonzero(out.max(axis=1))[0], np.nonzero(out.max(axis=0))[0]
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def s_resolution(content_box, pool: int, top_q: float) -> float:
    """Smallest nonzero contact strength: one pixel, one channel, one uint8 level (2/255 in [-1, 1]),
    averaged over the 3 channels, a pool x pool cell and the k top cells of ``contact_strength``."""
    r0, r1, c0, c1 = (int(v) for v in content_box)
    k = max(1, int(top_q * ((r1 - r0) // pool) * ((c1 - c0) // pool)))
    return (2.0 / 255.0) / (3 * pool * pool * k)


def _video_backend() -> str:
    """The decoder training picks (data_loader.create_torch_dataset): torchcodec if importable, else pyav."""
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401
    except Exception:  # noqa: BLE001
        return "pyav"
    return "torchcodec"


def episode_strengths(
    ds: _data.LocalLeRobotV3Dataset, ep: int, views: list[str], content_box, pool: int, top_q: float, chunk: int
) -> np.ndarray:
    """s (T, V) for one episode: every frame against the episode's frame 0, per view."""
    start, end = ds.episode_data_index[ep]
    indices = list(range(start, end))
    timestamps = ds._get_query_timestamps(0.0, {k: indices for k in views})  # noqa: SLF001
    s = np.zeros((end - start, len(views)), dtype=np.float32)
    for v, key in enumerate(views):
        tac_0 = None
        for c0 in range(0, len(indices), chunk):
            sub = {key: timestamps[key][c0 : c0 + chunk]}
            frames = ds._query_videos(start + c0, sub, ep)[key]  # noqa: SLF001
            if frames.ndim == 3:
                frames = frames[None]
            tac = model_tactile_frames(frames)
            if tac_0 is None:
                tac_0 = tac[:1]
            s[c0 : c0 + tac.shape[0], v] = _ir.contact_strength(tac, tac_0, content_box, pool, top_q).numpy()
    return s


_WORKER: dict = {}


def _init_worker(root: str, views: list[str], backend: str, tolerance_s: float, threads: int) -> None:
    torch.set_num_threads(threads)
    _WORKER["ds"] = _data.LocalLeRobotV3Dataset(root, tolerance_s=tolerance_s, video_backend=backend, video_keys=views)
    _WORKER["views"] = views


def _episode_job(job: tuple) -> tuple[int, np.ndarray]:
    ep, content_box, pool, top_q, chunk = job
    return ep, episode_strengths(_WORKER["ds"], ep, _WORKER["views"], content_box, pool, top_q, chunk)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold of a 1-D sample: the histogram split maximising the between-class variance.

    Splits through empty bins all score the same; the middle of that run is returned, so a gap
    between the two classes is cut in its middle rather than at its lower edge.
    """
    hist, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist).astype(np.float64)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = m0 / np.maximum(w0, 1)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    between = np.where((w0 > 0) & (w1 > 0), w0 * w1 * (mu0 - mu1) ** 2, -1.0)
    first = last = int(np.argmax(between))
    while last + 1 < between.size and between[last + 1] == between[first]:
        last += 1
    return float(0.5 * (edges[first + 1] + edges[last + 1]))


def _log_otsu(s: np.ndarray, resolution: float) -> float | None:
    """Otsu threshold of log s in s units; exact zeros enter the histogram at ``resolution``."""
    if not bool((s > 0).any()):
        return None
    log_s = np.log(np.maximum(s, resolution))
    if float(log_s.min()) == float(log_s.max()):
        return None
    return float(np.exp(otsu_threshold(log_s)))


def compute_statistics(
    strengths: dict[int, np.ndarray],
    views: list[str],
    *,
    n0: int,
    tau_floor: str | float,
    n_sigma: float = 3.0,
    start_mad: float = 5.0,
    resolution: float | None = None,
) -> dict:
    """Thresholds and P(n_a) from per-episode contact strengths {episode: (T, V)} (module docstring).

    ``resolution``: the smallest nonzero s (``s_resolution``; default: the model's default content
    box, pool 4, top-q 5%), where the s == 0 frames enter the Otsu histogram.
    """
    episodes = sorted(strengths)
    all_s = np.concatenate([strengths[e] for e in episodes], axis=0)            # (F, V)
    # Without each episode's frame 0: the baseline itself (s = 0 by construction, not a measurement).
    body = np.concatenate([strengths[e][1:] for e in episodes], axis=0)         # (F - E, V)
    n_views = len(views)
    if resolution is None:
        resolution = s_resolution(_ir.DEFAULT_CONTENT_BOX, 4, 0.05)

    otsu = _log_otsu(body.reshape(-1), resolution)
    # The same split over the positive s alone: the floor to use instead when the lower group of
    # positive s turns out to be free-space jitter rather than light contact (see the warning).
    positive = body[body > 0]
    otsu_positive = _log_otsu(positive, resolution) if positive.size > 1 else None
    zero_frac = [float((body[:, v] == 0).mean()) if body.size else 0.0 for v in range(n_views)]
    many_zeros = {_short(view): f"{frac:.0%}" for view, frac in zip(views, zero_frac, strict=True) if frac > 0.1}
    if many_zeros:
        logging.warning(
            "share of frames after the baseline with s == 0 exactly (bit-identical to the episode's first frame): "
            "%s. They are the no-contact class of the Otsu split, so tau_min (%s) sits in the gap above them and "
            "any gel change counts as contact. Check hist_log_s.svg: if positive s below the contacts is "
            "free-space jitter, pass a fixed --tau-floor above it (Otsu of the positive s alone: %s).",
            many_zeros, otsu, otsu_positive,
        )
    if str(tau_floor).lower() == "otsu":
        if otsu is None:
            logging.warning("Otsu threshold undefined (no spread in log s); tau_floor = 0")
        floor, floor_method = (otsu or 0.0), "otsu"
    else:
        floor, floor_method = float(tau_floor), "fixed"

    # Noise window: frames 1..N0 (frame 0 is the baseline itself, s = 0 by construction).
    window = {e: strengths[e][1 : n0 + 1] for e in episodes}
    if any(w.shape[0] == 0 for w in window.values()):
        raise ValueError("an episode has no frame after its baseline; cannot estimate the noise level")
    mu, sigma, tau_raw, tau, per_view, excluded = [], [], [], [], {}, {}
    for v, view in enumerate(views):
        start = np.array([float(window[e][:, v].mean()) for e in episodes])
        med = float(np.median(start))
        mad = float(np.median(np.abs(start - med)))
        start_thr = med + start_mad * mad
        keep = [e for e, x in zip(episodes, start, strict=True) if x <= start_thr]
        excluded[view] = [int(e) for e, x in zip(episodes, start, strict=True) if x > start_thr]
        noise = np.concatenate([window[e][:, v] for e in keep])
        m, sd = float(noise.mean()), float(noise.std())
        mu.append(m)
        sigma.append(sd)
        tau_raw.append(m + n_sigma * sd)
        tau.append(max(m + n_sigma * sd, floor))
        sv = all_s[:, v]
        per_view[view] = {
            "mu": m,
            "sigma": sd,
            "tau_raw": m + n_sigma * sd,
            "tau": tau[-1],
            "otsu": _log_otsu(body[:, v], resolution),
            "zero_frac": zero_frac[v],
            "start_median": med,
            "start_mad": mad,
            "start_threshold": start_thr,
            "n_noise_samples": int(noise.size),
            "n_excluded_episodes": len(excluded[view]),
            "frac_active": float((sv > tau[-1]).mean()),
            "s_quantiles": {q: float(np.quantile(sv, float(q))) for q in ("0.5", "0.9", "0.99")},
            "s_max": float(sv.max()),
        }

    active = all_s > np.asarray(tau, dtype=np.float32)[None, :]
    na_counts = np.bincount(active.sum(axis=1), minlength=n_views + 1)
    return {
        "tau": tau,
        "tau_floor": floor,
        "tau_floor_method": floor_method,
        "otsu_threshold": otsu,
        "otsu_positive_only": otsu_positive,
        "s_resolution": resolution,
        "zero_frac": zero_frac,
        "mu": mu,
        "sigma": sigma,
        "tau_raw": tau_raw,
        "n_sigma": n_sigma,
        "noise_frames": [1, n0],
        "start_mad_k": start_mad,
        "na_hist": (na_counts / na_counts.sum()).tolist(),
        "na_counts": na_counts.tolist(),
        "num_episodes": len(episodes),
        "num_frames": int(all_s.shape[0]),
        "per_view": per_view,
        "excluded_episodes": excluded,
    }


def rule_allocation(s: np.ndarray, tau: list[float], sibling: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(n, active), both (F, V): what the model's ``rule`` mode allocates for these strengths."""
    s_t = torch.from_numpy(np.ascontiguousarray(s, dtype=np.float32))
    n, active = _ir.allocate(
        s_t, torch.tensor(tau), "rule", torch.ones_like(s_t, dtype=torch.bool), sibling=sibling, return_active=True
    )
    return n.numpy(), active.numpy()


# ---------------------------------------------------------------------------
# SVG diagnostics (no matplotlib in the vtla env)
# ---------------------------------------------------------------------------
_SURFACE, _INK, _INK_2, _MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
_GRID, _AXIS = "#e1e0d9", "#c3c2b7"
_SERIES = "#2a78d6"  # one series per panel
# Ordinal ramp for the token count of a view (one hue, light -> dark). Level 2 is in the level set
# but no mode's table produces it.
_LEVEL_FILL = {1: "#86b6ef", 2: "#6da7ec", 5: "#5598e7", 10: "#2a78d6", 17: "#1c5cab", 26: "#104281"}


class _Svg:
    """Minimal SVG canvas."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self._items = [f'<rect width="{width}" height="{height}" fill="{_SURFACE}"/>']

    def line(self, x0, y0, x1, y1, color=_GRID, width=1.0):
        self._items.append(
            f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{color}" stroke-width="{width}"/>'
        )

    def rect(self, x, y, w, h, fill):
        self._items.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0.0):.1f}" height="{max(h, 0.0):.1f}" '
                           f'fill="{fill}"/>')

    def path(self, points, color=_SERIES, width=2.0):
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self._items.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
                           'stroke-linejoin="round" stroke-linecap="round"/>')

    def text(self, x, y, s, size=11, color=_INK_2, anchor="start", weight="normal"):
        self._items.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
                           f'text-anchor="{anchor}" font-weight="{weight}">{html.escape(str(s))}</text>')

    def save(self, path: pathlib.Path) -> None:
        body = "\n".join(self._items)
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" '
            f'font-family="system-ui, -apple-system, Segoe UI, sans-serif">\n{body}\n</svg>\n',
            encoding="utf-8",
        )


class _Panel:
    """Axes box mapping data to pixels, linear on both axes (log quantities are drawn as log10)."""

    def __init__(self, svg: _Svg, left, top, width, height, xlim, ylim):
        self.svg, self.left, self.top, self.width, self.height = svg, left, top, width, height
        self.xlim, self.ylim = xlim, ylim

    def x(self, v: float) -> float:
        x0, x1 = self.xlim
        return self.left + (min(max(v, x0), x1) - x0) / (x1 - x0) * self.width

    def y(self, v: float) -> float:
        y0, y1 = self.ylim
        return self.top + self.height - (min(max(v, y0), y1) - y0) / (y1 - y0) * self.height

    def axes(self, xticks, yticks, xfmt=str, yfmt=str, *, title=None, xlabel=None, show_xticks=True):
        for t in yticks:
            self.svg.line(self.left, self.y(t), self.left + self.width, self.y(t), _GRID)
            self.svg.text(self.left - 6, self.y(t) + 4, yfmt(t), size=10, color=_MUTED, anchor="end")
        self.svg.line(self.left, self.top + self.height, self.left + self.width, self.top + self.height, _AXIS)
        if show_xticks:
            for t in xticks:
                self.svg.text(self.x(t), self.top + self.height + 14, xfmt(t), size=10, color=_MUTED, anchor="middle")
        if title:
            self.svg.text(self.left, self.top - 8, title, size=12, color=_INK, weight="600")
        if xlabel:
            self.svg.text(self.left + self.width / 2, self.top + self.height + 30, xlabel, size=11, anchor="middle")

    def vline(self, v: float, color: str, label: str, label_row: int) -> None:
        """A labelled reference line; labels are stacked in rows so close lines do not collide."""
        x = self.x(v)
        self.svg.line(x, self.top, x, self.top + self.height, color, 1.5)
        self.svg.text(x + 4, self.top + 12 + 13 * label_row, label, size=10, color=_INK_2)


def _decade_ticks(lo: float, hi: float) -> list[int]:
    return list(range(math.ceil(lo), math.floor(hi) + 1))


def _nice_ticks(hi: float, target: int = 5) -> list[float]:
    """0, step, 2 step, ... <= hi with step in {1, 2, 5} x 10^k (about ``target`` ticks)."""
    if hi <= 0:
        return [0]
    raw = hi / target
    base = 10 ** math.floor(math.log10(raw))
    step = next(m * base for m in (1, 2, 5, 10) if m * base >= raw)
    step = max(step, 1) if hi >= 1 else step
    return [i * step for i in range(int(hi // step) + 1)]


def _fmt_decade(t: float) -> str:
    return f"1e{int(t)}"


def _short(view: str) -> str:
    return view.rsplit(".", 1)[-1]


def _log10_range(all_s: np.ndarray, marks=()) -> tuple[float, float]:
    """Decade range of log10 s: from the 0.1% quantile of the positive s to the maximum, widened to
    include the positive ``marks`` (thresholds) so their lines land where they are."""
    pos = all_s[all_s > 0]
    marks = [float(m) for m in marks if 0 < float(m) < math.inf]
    lows = ([float(np.quantile(pos, 0.001))] if pos.size else []) + marks
    highs = ([float(pos.max())] if pos.size else []) + marks
    if not lows:
        return -4.0, 0.0
    lo = math.floor(math.log10(min(lows)))
    return lo, max(math.ceil(math.log10(max(highs))), lo + 1)


def plot_histograms(path: pathlib.Path, strengths: dict[int, np.ndarray], views: list[str], stats: dict) -> None:
    """Per-view histogram of log10 s over all frames after the baselines with the three thresholds;
    the frames with s == 0 exactly (no log) are the separate bar at "0", left of the log axis."""
    body = np.concatenate([strengths[e][1:] for e in sorted(strengths)], axis=0)
    lo, hi = _log10_range(body, [stats["tau_floor"], *stats["tau"]])
    edges = np.linspace(lo, hi, max(60, 20 * (hi - lo)) + 1)
    zero_w = 0.15 * (hi - lo)  # width of the s == 0 slot in log10 units
    cols, pw, ph = 2, 420, 200
    rows = math.ceil(len(views) / cols)
    svg = _Svg(cols * (pw + 90) + 20, rows * (ph + 90) + 50)
    svg.text(20, 26, "Contact strength per view: log10 s after the baseline frame, with mu + 3 sigma, tau_min and tau",
             size=14, color=_INK, weight="600")
    for v, view in enumerate(views):
        sv = body[:, v]
        counts, _ = np.histogram(np.log10(sv[sv > 0]), bins=edges)
        n_zero = int((sv == 0).sum())
        top = max(int(counts.max()), 1)
        panel = _Panel(svg, 70 + (v % cols) * (pw + 90), 70 + (v // cols) * (ph + 90), pw, ph,
                       (lo - zero_w, hi), (0, top))
        panel.axes(_decade_ticks(lo, hi), _nice_ticks(top, 4), _fmt_decade, lambda t: f"{t:g}",
                   title=f"{_short(view)}: frames per bin; s = 0 (bar at 0): {n_zero} frames, "
                         f"{stats['per_view'][view]['zero_frac']:.0%}",
                   xlabel="s (log scale)")
        svg.text(panel.x(lo - 0.65 * zero_w), panel.top + panel.height + 14, "0", size=10, color=_MUTED,
                 anchor="middle")
        if n_zero:  # clipped at the top of the axis when it outgrows the histogram (count in the title)
            x0, x1 = panel.x(lo - 0.9 * zero_w), panel.x(lo - 0.4 * zero_w)
            svg.rect(x0, panel.y(min(n_zero, top)), x1 - x0, panel.y(0) - panel.y(min(n_zero, top)), _MUTED)
        for c, a, b in zip(counts, edges[:-1], edges[1:], strict=True):
            if c:
                svg.rect(panel.x(a) + 0.5, panel.y(c), panel.x(b) - panel.x(a) - 1.0, panel.y(0) - panel.y(c), _SERIES)
        pv = stats["per_view"][view]
        if pv["tau_raw"] > 0:
            panel.vline(math.log10(pv["tau_raw"]), _INK_2, f"mu+{stats['n_sigma']:g}sigma {pv['tau_raw']:.3g}", 1)
        if stats["tau_floor"] > 0:
            panel.vline(math.log10(stats["tau_floor"]), _MUTED, f"tau_min {stats['tau_floor']:.3g}", 2)
        if pv["tau"] > 0:
            panel.vline(math.log10(pv["tau"]), _INK, f"tau {pv['tau']:.3g}", 0)
    svg.save(path)


def plot_episode(
    path: pathlib.Path, ep: int, s: np.ndarray, n: np.ndarray, views: list[str], stats: dict, fps: float, ylim
) -> None:
    """s_k(t) per view with tau_k (small multiples), then the rule allocation n_k(t) as level strips."""
    T = s.shape[0]
    pw, ph, gap = 860, 92, 34
    strip_top = 60 + len(views) * (ph + gap) + 26
    svg = _Svg(90 + pw + 100, strip_top + len(views) * 18 + 90)
    svg.text(20, 26, f"Episode {ep}: contact strength s_k(t) and threshold tau_k ({T} frames at {fps:g} fps)",
             size=14, color=_INK, weight="600")
    xticks = [int(t) for t in _nice_ticks(max(T - 1, 1), 8)]
    for v, view in enumerate(views):
        panel = _Panel(svg, 90, 60 + v * (ph + gap), pw, ph, (0, max(T - 1, 1)), ylim)
        panel.axes(xticks, _decade_ticks(*ylim), str, _fmt_decade, title=f"{_short(view)}: s (log scale)",
                   xlabel="frame" if v == len(views) - 1 else None, show_xticks=v == len(views) - 1)
        logs = np.log10(np.maximum(s[:, v], 10.0 ** ylim[0]))
        svg.path([(panel.x(t), panel.y(float(y))) for t, y in enumerate(logs)])
        tau = stats["tau"][v]
        if tau > 0:
            y = panel.y(math.log10(tau))
            svg.line(panel.left, y, panel.left + panel.width, y, _INK, 1.5)
            svg.text(panel.left + panel.width + 6, y + 4, f"tau {tau:.3g}", size=10, color=_INK_2)
    svg.text(90, strip_top - 4, "Rule allocation n_k(t): tokens per view", size=12, color=_INK, weight="600")
    for v, view in enumerate(views):
        y = strip_top + 6 + v * 18
        svg.text(84, y + 11, _short(view), size=10, color=_MUTED, anchor="end")
        t0 = 0
        for t in range(1, T + 1):
            if t == T or n[t, v] != n[t0, v]:
                x0 = 90 + t0 / max(T, 1) * pw
                x1 = 90 + t / max(T, 1) * pw
                svg.rect(x0, y, x1 - x0, 14, _LEVEL_FILL.get(int(n[t0, v]), _SURFACE))
                t0 = t
    legend_y = strip_top + 6 + len(views) * 18 + 22
    svg.text(90, legend_y + 10, "tokens:", size=10, color=_INK_2)
    for i, level in enumerate(lv for lv in _ir.LEVEL_TOKENS if lv != 2):
        x = 140 + i * 56
        svg.rect(x, legend_y, 14, 12, _LEVEL_FILL[level])
        svg.text(x + 19, legend_y + 10, str(level), size=10, color=_INK_2)
    svg.save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(levelname)s %(message)s")
    root = pathlib.Path(args.repo_id)
    if not _data._is_local_lerobot_v3_root(root):  # noqa: SLF001
        raise SystemExit(f"{root} is not a local LeRobot dataset root (meta/info.json, data/, tasks, episodes)")
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info["features"]
    views = args.tactile_keys or sorted(
        k for k, ft in features.items() if ft.get("dtype") == "video" and "tactile" in k
    )
    missing = [k for k in views if features.get(k, {}).get("dtype") != "video"]
    if not views or missing:
        raise SystemExit(f"no usable tactile video columns (requested {views}, not video: {missing})")
    if len(args.sibling) != len(views):
        raise SystemExit(f"--sibling needs one entry per view ({len(views)}), got {args.sibling}")

    ft = features[views[0]]
    names = list(ft.get("names") or [])
    shape = list(ft["shape"])
    if "height" in names and "width" in names:
        height, width = shape[names.index("height")], shape[names.index("width")]
    else:
        height, width = (shape[1], shape[2]) if shape[0] == 3 else (shape[0], shape[1])
    auto_box = letterbox_content_box(int(height), int(width))
    content_box = tuple(args.content_box) if args.content_box else auto_box
    if content_box != auto_box:
        logging.warning("--content-box %s differs from the letterbox content %s of %dx%d frames",
                        content_box, auto_box, height, width)
    logging.info("views=%s frames=%dx%d content_box=%s pool=%d top_q=%g n0=%d",
                 views, height, width, content_box, args.pool, args.top_q, args.n0)

    backend = _video_backend()
    ds = _data.LocalLeRobotV3Dataset(root, tolerance_s=args.tolerance_s, video_backend=backend, video_keys=views)
    episodes = sorted(ds.episode_data_index)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    logging.info("decoding %d episodes with %s (%d workers)", len(episodes), backend, args.num_workers)

    t_start = time.time()
    jobs = [(ep, content_box, args.pool, args.top_q, args.chunk) for ep in episodes]
    strengths: dict[int, np.ndarray] = {}
    if args.num_workers > 0:
        threads = max(1, (os.cpu_count() or 1) // args.num_workers)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(args.num_workers, _init_worker, (str(root), views, backend, args.tolerance_s, threads)) as pool:
            for i, (ep, s) in enumerate(pool.imap_unordered(_episode_job, jobs), 1):
                strengths[ep] = s
                if i % 10 == 0 or i == len(jobs):
                    logging.info("%d/%d episodes (%.0fs)", i, len(jobs), time.time() - t_start)
    else:
        for i, (ep, *rest) in enumerate(jobs, 1):
            strengths[ep] = episode_strengths(ds, ep, views, *rest)
            if i % 10 == 0 or i == len(jobs):
                logging.info("%d/%d episodes (%.0fs)", i, len(jobs), time.time() - t_start)

    stats = compute_statistics(
        strengths, views, n0=args.n0, tau_floor=args.tau_floor, n_sigma=args.n_sigma, start_mad=args.start_mad,
        resolution=s_resolution(content_box, args.pool, args.top_q),
    )
    all_s = np.concatenate([strengths[e] for e in sorted(strengths)], axis=0)
    levels, _ = rule_allocation(all_s, stats["tau"], args.sibling)
    level_frac = {
        _short(view): {str(lv): float((levels[:, v] == lv).mean()) for lv in _ir.LEVEL_TOKENS}
        for v, view in enumerate(views)
    }
    out = {
        "version": 1,
        "repo_id": str(root.resolve()),  # compared with the training data path by the trainer
        "views": views,
        "frame_size": [int(height), int(width)],
        "image_resolution": list(_model.IMAGE_RESOLUTION),
        "content_box": list(content_box),
        "topq": args.top_q,
        "pool": args.pool,
        "n0": args.n0,
        **stats,
        "rule_allocation": {
            "sibling": list(args.sibling),
            "level_frac": level_frac,
            "mean_tokens": levels.mean(axis=0).tolist(),
        },
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"\n{len(strengths)} episodes, {stats['num_frames']} frames; tau_floor={stats['tau_floor']:.4g} "
          f"({stats['tau_floor_method']}; Otsu on log s = {stats['otsu_threshold']}, s = 0 entered at "
          f"{stats['s_resolution']:.3g})")
    print(f"{'view':<34} {'mu':>10} {'sigma':>10} {'mu+3sig':>10} {'tau':>10} {'active':>8} {'s=0':>6} {'excl':>5}")
    for v, view in enumerate(views):
        pv = stats["per_view"][view]
        print(f"{view:<34} {pv['mu']:>10.4g} {pv['sigma']:>10.4g} {pv['tau_raw']:>10.4g} {pv['tau']:>10.4g} "
              f"{pv['frac_active']:>8.3f} {pv['zero_frac']:>6.3f} {pv['n_excluded_episodes']:>5d}")
    print(f"P(n_a) = {[round(x, 4) for x in stats['na_hist']]}  (P(n_a >= 1) = {1 - stats['na_hist'][0]:.4f})")
    print(f"rule: mean tokens per view = {[round(x, 2) for x in out['rule_allocation']['mean_tokens']]}")
    print(f"wrote {out_path}")

    if args.plot_dir:
        plot_dir = pathlib.Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_histograms(plot_dir / "hist_log_s.svg", strengths, views, stats)
        ylim = _log10_range(all_s, stats["tau"])
        order = sorted(strengths)
        picks = sorted({order[int(i)] for i in np.linspace(0, len(order) - 1, min(args.plot_episodes, len(order)))})
        for ep in picks:
            n_ep, _ = rule_allocation(strengths[ep], stats["tau"], args.sibling)
            plot_episode(plot_dir / f"episode_{ep}.svg", ep, strengths[ep], n_ep, views, stats, ds.fps, ylim)
        print(f"wrote {len(picks) + 1} plots to {plot_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
