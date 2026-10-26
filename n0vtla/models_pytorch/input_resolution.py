"""1A input-side dynamic resolution: per-view tactile token counts driven by contact strength.

The original predictor path (``n0vtla_policy.py::_build_g``) hands every tactile view to the
predictor as the same 10 tokens (DINOv2 CLS + 3x3 pooled patch grid), contact or not. 1A keeps the
frozen DINOv2 encoder, ``tactile_proj`` and the predictor unchanged and only decides HOW MANY
tokens each view contributes, within the same budget of 40:

  (1) s_k   = contact_strength(tac_t^k, tac_0^k)       pixel level, no learning, no grad
  (2) a_k   = [s_k > max(tau_k, tau_floor)]             thresholds from scripts/tactile_contact_stats.py
  (3) n_k   = allocate(s, a)                            n_k in {1, 2, 5, 10, 17, 26}, sum <= 40
  (4) tok_k = [CLS ; AdaptiveAvgPool_g(P_k)]            g = sqrt(n_k - 1); ONE DINOv2 pass per view
                                                        (FrozenDINOv2TactileEncoder.encode_backbone_multi)
  (5) x_kj  = tactile_proj(tok_kj) + e_view[k] + e_level[n_k]

The token count is the knob that matters: the predictor's cross-attention layer-norms every key /
value (``norm_kv``), so scaling a view's tokens is a no-op, while giving a view n_k tokens both adds
spatial detail and adds ~log(n_k) to its share of the attention logits.

The predictor input keeps a FIXED layout: ``MAX_TOKENS_PER_VIEW`` (26) slots per view in
``tactile_image_keys`` order, of which view k fills the first n_k; ``g_mask`` marks exactly those
slots, so the masked cross-attention sees the n_k valid tokens and nothing else. With ``uniform10``
and both embeddings at their zero init, the valid tokens are bit-identical to the original g; only
the attention reduction runs over a longer, masked key set (float32-rounding-level difference).

The allocator is a fixed rule (no learnable parameters); it and the contact strength run under
``torch.no_grad()``. The only new trainable parameters are the view and level embeddings (zero
init, so the first step reproduces the loaded model up to the changed token counts).

Modes (``N0VTLAConfig.input_res_mode``; "off" never constructs this module):
  * ``uniform10``: 10 tokens per view (equivalence check against the original g)
  * ``uniform5``:  5 tokens per view (only fewer tokens)
  * ``random``:    n_a ~ P(n_a), random active views, random ranking; same table as ``rule``
  * ``rule``:      the contact rule below (the 1A method)
  * ``rule_var``:  10 tokens per active view, 1 per inactive view (variable budget, 4-40)

Allocation table for ``rule`` / ``random`` (n_a = number of active valid views; "sibling" is the
other finger of the same gripper, ``input_res_sibling``):

  n_a = 0 or 4  every valid view 10                                  (40, same as the original)
  n_a = 1       active 26, its sibling 10, every other view 1        (38)
  n_a = 2       both active 17, stronger inactive 5, weaker inactive 1 (40)
  n_a = 3       strongest active 17, other actives 10, inactive 1   (38)

A view that is absent or a placeholder (``image_mask=False``) is "invalid": it gets 0 tokens, is
never active and takes no part in the ranking; the row is still chosen by n_a and the slots it
would have taken are simply not used, so every valid view keeps >= 1 token and the total stays
<= 40. Ties in s keep the ``tactile_image_keys`` order.

Statistics and checkpoints: the thresholds / P(n_a) come from the config or the stats JSON
(``input_res_stats``) and are saved with the model, so a trained 1A checkpoint serves on its own.
A model may be built without them; it then refuses to allocate until a checkpoint provides them.
The mode is saved too: loading a 1A checkpoint under another mode, or under a config with 1A off,
is an error (``split_checkpoint_keys``, ``_check_loaded_statistics``).
"""
from __future__ import annotations

from collections.abc import Sequence
import json
import logging
import os
import pathlib
from typing import Literal, get_args

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

# Every input_res parameter / buffer lives under this state_dict prefix. Checkpoints trained
# without 1A (n0-vtla-base, the released sim checkpoints, B0) lack these keys; the loaders accept
# exactly this prefix as missing and the module keeps its initial values.
INPUT_RES_PREFIX = "input_res."

# N0VTLAConfig.input_res_mode (a Literal, so the CLI rejects a misspelt mode).
AllocationMode = Literal["off", "uniform10", "uniform5", "random", "rule", "rule_var"]
ALLOCATION_MODES = get_args(AllocationMode)
# Modes that need the per-view thresholds tau, and the mode that needs P(n_a).
THRESHOLD_MODES = ("rule", "rule_var")
NA_HIST_MODES = ("random",)

# Token count per view -> pooling grid g (n = 1 + g^2; g = 0 means CLS only).
LEVEL_TOKENS = (1, 2, 5, 10, 17, 26)
TOKENS_TO_GRID = {1: 0, 2: 1, 5: 2, 10: 3, 17: 4, 26: 5}
LEVEL_INDEX = {n: i for i, n in enumerate(LEVEL_TOKENS)}
MAX_TOKENS_PER_VIEW = max(LEVEL_TOKENS)
TOKEN_BUDGET = 40
# The table above is written for at most 4 views (two grippers x two fingers).
MAX_VIEWS = 4

# Real robot: 700x400 portrait tactile frames letterboxed to 224x224 -> content columns 48-176.
# NeoSim: 240x320 landscape frames -> content rows 28-196 (the sim configs override this).
DEFAULT_CONTENT_BOX = (0, 224, 48, 176)
DEFAULT_SIBLING = (1, 0, 3, 2)


# ---------------------------------------------------------------------------
# (1) contact strength
# ---------------------------------------------------------------------------
@torch.no_grad()
def contact_strength(
    tac_t: torch.Tensor,
    tac_0: torch.Tensor,
    content_box: Sequence[int] = DEFAULT_CONTENT_BOX,
    pool: int = 4,
    top_q: float = 0.05,
) -> torch.Tensor:
    """Per-sample contact strength s (B,) of one tactile view.

    ``tac_t`` / ``tac_0``: (B, C, H, W) current / episode-start frames exactly as the model sees them
    (letterboxed to 224x224, values in [-1, 1]); ``tac_0`` may have batch 1 and is broadcast.

        s = mean( top_q( AvgPool_pool( mean_c |tac_t - tac_0| restricted to content_box ) ) )

    * ``content_box`` = (r0, r1, c0, c1): the letterbox bars are identical in both frames, so their
      difference is always 0 and would only dilute the statistic;
    * ``pool`` x ``pool`` average pooling suppresses isolated noisy pixels (a remainder that does
      not fill a whole cell is dropped);
    * ``top_q``: mean over the largest ``top_q`` fraction of cells (at least one), so a small
      contact patch is not averaged away by the untouched part of the gel.
    """
    if tac_t.ndim != 4 or tac_t.shape[1] not in (1, 3):
        raise ValueError(f"contact_strength expects (B, C, H, W) tactile tensors, got {tuple(tac_t.shape)}")
    r0, r1, c0, c1 = (int(v) for v in content_box)
    if not (0 <= r0 < r1 <= tac_t.shape[2] and 0 <= c0 < c1 <= tac_t.shape[3]):
        raise ValueError(f"content_box {tuple(content_box)} does not fit frames of size {tuple(tac_t.shape[2:])}")
    d = (tac_t.float() - tac_0.float()).abs().mean(dim=1)     # (B, H, W) channel-mean |D|
    d = d[:, r0:r1, c0:c1]                                     # content box only
    d = F.avg_pool2d(d[:, None], kernel_size=pool)[:, 0]       # (B, h // pool, w // pool)
    flat = d.flatten(1)
    k = max(1, int(top_q * flat.shape[1]))
    return flat.topk(k, dim=1).values.mean(dim=1)


# ---------------------------------------------------------------------------
# (2)-(3) activity + allocation
# ---------------------------------------------------------------------------
def _rule_row(
    order: Sequence[int], active: Sequence[bool], valid: Sequence[bool], sibling: Sequence[int]
) -> list[int]:
    """Token counts of one sample from the allocation table (module docstring).

    ``order`` ranks the views (strongest first); invalid views get 0 and are skipped.
    """
    acts = [k for k in order if valid[k] and active[k]]
    inacts = [k for k in order if valid[k] and not active[k]]
    n = [0] * len(valid)
    if len(acts) in (0, 4):
        for k in acts + inacts:
            n[k] = 10
    elif len(acts) == 1:
        a = acts[0]
        for k in inacts:
            n[k] = 1
        n[a] = 26
        if valid[sibling[a]]:
            n[sibling[a]] = 10
    elif len(acts) == 2:
        for k in acts:
            n[k] = 17
        for i, k in enumerate(inacts):
            n[k] = 5 if i == 0 else 1
    else:
        for i, k in enumerate(acts):
            n[k] = 17 if i == 0 else 10
        for k in inacts:
            n[k] = 1
    return n


@torch.no_grad()
def allocate(
    s: torch.Tensor,
    threshold: torch.Tensor | None,
    mode: str,
    valid: torch.Tensor,
    *,
    sibling: Sequence[int] = DEFAULT_SIBLING,
    na_hist: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    return_active: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Per-view token counts n (B, V) long on the CPU; 0 for invalid views.

    s: (B, V) contact strengths; threshold: (V,) effective thresholds (required by ``rule`` /
    ``rule_var``; for the uniform modes it only feeds the returned active flags); valid: (B, V)
    bool; na_hist: (V + 1,) P(n_a) (``random``). ``random`` draws from ``generator`` (default:
    torch's global CPU generator, which the trainer seeds per rank). With ``return_active`` also
    returns the (B, V) bool active flags.
    """
    if mode not in ALLOCATION_MODES or mode == "off":
        raise ValueError(f"unknown input_res mode {mode!r}; expected one of {ALLOCATION_MODES[1:]}")
    if s.ndim != 2 or valid.shape != s.shape:
        raise ValueError(f"allocate expects s and valid of shape (B, V), got {tuple(s.shape)} / {tuple(valid.shape)}")
    batch, n_views = s.shape
    if n_views > MAX_VIEWS or len(sibling) != n_views:
        raise ValueError(f"allocate supports 1..{MAX_VIEWS} views with one sibling each, got {n_views} / {sibling}")
    s = s.detach().float().cpu()
    valid = valid.detach().to(torch.bool).cpu()
    n = torch.zeros(batch, n_views, dtype=torch.long)

    orders: list[list[int]] = []
    if mode == "random":
        probs = None if na_hist is None else na_hist.detach().float().cpu()
        if probs is None or probs.numel() != n_views + 1 or bool((probs < 0).any()) or float(probs.sum()) <= 0:
            raise ValueError(f"mode 'random' needs P(n_a) with {n_views + 1} non-negative entries, got {na_hist}")
        n_active = torch.multinomial(probs, batch, replacement=True, generator=generator)
        active = torch.zeros_like(valid)
        for b in range(batch):
            idx = valid[b].nonzero().flatten()
            k = min(int(n_active[b]), idx.numel())
            if k:
                active[b, idx[torch.randperm(idx.numel(), generator=generator)[:k]]] = True
            # A random ranking as well: which active view gets 17 and which inactive view gets 5
            # must not leak the contact strength into the control.
            orders.append(torch.randperm(n_views, generator=generator).tolist())
    elif threshold is not None:
        active = (s > threshold.detach().float().cpu().reshape(1, n_views)) & valid
    elif mode in THRESHOLD_MODES:
        raise ValueError(f"mode {mode!r} needs per-view thresholds")
    else:
        active = torch.zeros_like(valid)

    if mode in ("uniform10", "uniform5"):
        n[valid] = 10 if mode == "uniform10" else 5
        return (n, active) if return_active else n
    if mode == "rule_var":
        n[valid] = 1
        n[active] = 10
        return (n, active) if return_active else n

    if mode == "rule":
        # Strongest first; Python's sort is stable, so ties keep the view order.
        orders = [sorted(range(n_views), key=lambda k, row=row: -row[k]) for row in s.tolist()]
    valid_rows, active_rows = valid.tolist(), active.tolist()
    for b in range(batch):
        n[b] = torch.tensor(_rule_row(orders[b], active_rows[b], valid_rows[b], sibling), dtype=torch.long)
    return (n, active) if return_active else n


# ---------------------------------------------------------------------------
# Statistics written by scripts/tactile_contact_stats.py
# ---------------------------------------------------------------------------
def load_contact_stats(path: str | pathlib.Path) -> dict:
    """Read the JSON written by ``scripts/tactile_contact_stats.py`` (fails loudly if missing)."""
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"1A contact statistics not found: {path}. Run scripts/tactile_contact_stats.py and point "
            "VTLA_1A_STATS (input_res_stats) at its --out JSON."
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _short_view_name(key: str) -> str:
    """`observation.images.tactile_a` -> `tactile_a` (the model-side view name)."""
    return key.rsplit(".", 1)[-1]


def split_checkpoint_keys(
    model: nn.Module, missing: Sequence[str], unexpected: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Set aside the 1A keys of a ``strict=False`` load: (other missing keys, missing ``input_res.*``).

    ``input_res.*`` keys the checkpoint lacks are expected when it was trained without 1A
    (n0-vtla-base, B0, the released checkpoints); the module then keeps its initial values and the
    caller decides whether that is acceptable. ``input_res.*`` keys the model does not have mean the
    checkpoint WAS trained with 1A: loading it under a config with 1A off would silently drop the
    trained allocation and embeddings, so that raises.
    """
    extra = sorted(k for k in unexpected if k.startswith(INPUT_RES_PREFIX))
    if extra:
        mode = getattr(getattr(model, "config", None), "input_res_mode", "off")
        raise ValueError(
            f"the checkpoint holds 1A weights {extra} that the model (input_res_mode={mode!r}) does not have: "
            "load a 1A checkpoint with the config it was trained with"
        )
    return (
        [k for k in missing if not k.startswith(INPUT_RES_PREFIX)],
        sorted(k for k in missing if k.startswith(INPUT_RES_PREFIX)),
    )


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
class InputResolution(nn.Module):
    """1A allocator state plus the only new trainable parameters (view / level embeddings).

    Buffers (statistics, not weights; persistent, so a trained 1A checkpoint carries the thresholds
    it was trained with and serving reproduces the training-time allocation):
      * ``tau``       (V,)    per-view threshold (mu + 3 sigma, already floored by the stats script);
                              +inf when unset (modes that do not use it, or not loaded yet)
      * ``tau_floor`` ()      lower bound applied at run time: threshold = max(tau, tau_floor)
      * ``na_hist``   (V + 1,) P(n_a), used by ``random`` (all zero when unset)
      * ``mode_id``   ()      index of the mode in ALLOCATION_MODES (a checkpoint loads only into its mode)
    """

    def __init__(
        self,
        llm_dim: int,
        n_views: int,
        mode: str,
        *,
        content_box: Sequence[int] = DEFAULT_CONTENT_BOX,
        top_q: float = 0.05,
        pool: int = 4,
        tau: Sequence[float] | None = None,
        tau_floor: float = 0.0,
        na_hist: Sequence[float] | None = None,
        sibling: Sequence[int] = DEFAULT_SIBLING,
    ) -> None:
        super().__init__()
        if mode not in ALLOCATION_MODES or mode == "off":
            raise ValueError(f"unknown input_res mode {mode!r}; expected one of {ALLOCATION_MODES[1:]}")
        if not 1 <= n_views <= MAX_VIEWS:
            raise ValueError(f"1A supports 1..{MAX_VIEWS} tactile views, got {n_views}")
        sibling = tuple(int(v) for v in sibling)
        if len(sibling) != n_views or any(not 0 <= v < n_views or v == k for k, v in enumerate(sibling)):
            raise ValueError(f"input_res_sibling {sibling} is not a valid other-finger map for {n_views} views")
        content_box = tuple(int(v) for v in content_box)
        if len(content_box) != 4 or not (content_box[0] < content_box[1] and content_box[2] < content_box[3]):
            raise ValueError(f"input_res_content_box must be (r0, r1, c0, c1), got {content_box}")
        if not 0.0 < float(top_q) <= 1.0 or int(pool) < 1:
            raise ValueError(f"need 0 < input_res_topq <= 1 and input_res_pool >= 1, got {top_q} / {pool}")
        # Statistics may be absent here (serving a trained checkpoint, which carries them);
        # missing_statistics() reports that, and allocate() refuses to run without them.
        if tau is not None and len(tau) != n_views:
            raise ValueError(f"input_res_mode={mode!r} needs {n_views} thresholds tau, got {tau}")
        if na_hist is not None and (len(na_hist) != n_views + 1 or min(na_hist) < 0):
            raise ValueError(f"input_res_mode={mode!r} needs P(n_a) with {n_views + 1} entries, got {na_hist}")

        self.mode = mode
        self.n_views = n_views
        self.content_box = content_box
        self.top_q = float(top_q)
        self.pool = int(pool)
        self.sibling = sibling
        # Zero init: adding them is a no-op at step 0 (uniform10 then reproduces the original g).
        self.view_embed = nn.Parameter(torch.zeros(n_views, llm_dim))
        self.level_embed = nn.Parameter(torch.zeros(len(LEVEL_TOKENS), llm_dim))
        tau_t = torch.full((n_views,), float("inf")) if tau is None else torch.tensor([float(v) for v in tau])
        self.register_buffer("tau", tau_t)
        self.register_buffer("tau_floor", torch.tensor(float(tau_floor)))
        na_t = torch.zeros(n_views + 1) if na_hist is None else torch.tensor([float(v) for v in na_hist])
        self.register_buffer("na_hist", na_t / na_t.sum() if float(na_t.sum()) > 0 else na_t)
        self.register_buffer("mode_id", torch.tensor(ALLOCATION_MODES.index(mode)))
        self.register_load_state_dict_pre_hook(_check_loaded_statistics)
        # Where the configured statistics came from (from_config); for the dataset check of the trainer.
        self.stats_path: str | None = None
        self.stats_repo_id: str | None = None

    @classmethod
    def from_config(cls, config, llm_dim: int) -> InputResolution:
        """Build from ``N0VTLAConfig.input_res_*``; explicit fields win over the stats JSON.

        ``input_res_stats`` (the sim configs read it from ``VTLA_1A_STATS``, else the S6 output
        under assets/) names the JSON written by ``scripts/tactile_contact_stats.py``. A set path
        that does not exist, or statistics computed with a different contact-strength definition
        (content box / top-q / pool / views) fail here, at model construction. A mode whose
        statistics nobody provided is still built (a trained 1A checkpoint carries them); the
        loaders and ``allocate`` check ``missing_statistics()`` once the weights are in.
        """
        mode = str(config.input_res_mode)
        views = tuple(getattr(config, "tactile_image_keys", ()) or ())
        stats_path = getattr(config, "input_res_stats", None)
        stats = load_contact_stats(stats_path) if stats_path else {}
        content_box = tuple(int(v) for v in config.input_res_content_box)
        if stats:
            expected = {"content_box": list(content_box), "topq": float(config.input_res_topq),
                        "pool": int(config.input_res_pool)}
            got = {"content_box": [int(v) for v in stats.get("content_box", [])],
                   "topq": float(stats.get("topq", float("nan"))), "pool": int(stats.get("pool", -1))}
            if got != expected:
                raise ValueError(
                    f"{stats_path} was computed with {got} but the config uses {expected}; the "
                    "thresholds only hold for the same contact-strength definition"
                )
            stat_views = [_short_view_name(k) for k in stats.get("views", [])]
            if stat_views and stat_views != list(views):
                raise ValueError(
                    f"{stats_path} lists views {stat_views} but tactile_image_keys is {list(views)} "
                    "(per-view thresholds must follow the model's view order)"
                )
        tau = config.input_res_tau if config.input_res_tau is not None else stats.get("tau")
        tau_floor = config.input_res_tau_floor
        if tau_floor is None:
            tau_floor = stats.get("tau_floor", 0.0)
        na_hist = config.input_res_na_hist if config.input_res_na_hist is not None else stats.get("na_hist")
        module = cls(
            llm_dim,
            len(views),
            mode,
            content_box=content_box,
            top_q=config.input_res_topq,
            pool=config.input_res_pool,
            tau=tau,
            tau_floor=tau_floor,
            na_hist=na_hist,
            sibling=config.input_res_sibling,
        )
        module.stats_path, module.stats_repo_id = stats_path, stats.get("repo_id")
        logging.info(
            "1A input resolution: mode=%s views=%s threshold=%s na_hist=%s stats=%s",
            mode, list(views), [round(float(v), 6) for v in module.threshold], module.na_hist.tolist(),
            stats_path,
        )
        if module.missing_statistics():
            logging.info("1A input resolution: no contact statistics configured; they must come from the checkpoint")
        return module

    def extra_repr(self) -> str:
        return (f"mode={self.mode}, n_views={self.n_views}, content_box={self.content_box}, "
                f"top_q={self.top_q}, pool={self.pool}, sibling={self.sibling}")

    @property
    def threshold(self) -> torch.Tensor:
        """Effective per-view threshold (V,): max(tau_k, tau_floor)."""
        return torch.maximum(self.tau, self.tau_floor)

    def missing_statistics(self) -> str | None:
        """Why this mode cannot allocate yet (thresholds / P(n_a) never set), or None when it can."""
        if self.mode in THRESHOLD_MODES and not bool(torch.isfinite(self.tau).all()):
            what = f"thresholds tau (got {self.tau.tolist()})"
        elif self.mode in NA_HIST_MODES and float(self.na_hist.sum()) <= 0:
            what = "P(n_a)"
        else:
            return None
        return (
            f"input_res_mode={self.mode!r} has no {what}: set VTLA_1A_STATS (input_res_stats) to the JSON "
            "written by scripts/tactile_contact_stats.py, or load a checkpoint trained in this mode"
        )

    def stats_dataset_mismatch(self, repo_id: str | None) -> str | None:
        """A warning when the configured statistics were computed on another dataset than ``repo_id``."""
        if not (self.stats_repo_id and isinstance(repo_id, str) and repo_id):
            return None
        if os.path.realpath(self.stats_repo_id) == os.path.realpath(repo_id):
            return None
        return (
            f"1A contact statistics {self.stats_path} were computed on {self.stats_repo_id}, but the data is "
            f"{repo_id}: fine for statistics shared across tasks, wrong for a stale VTLA_1A_STATS"
        )

    @torch.no_grad()
    def contact_strength(self, tac_t: torch.Tensor, tac_0: torch.Tensor) -> torch.Tensor:
        """s (B,) of one view with this module's content box / pool / top-q."""
        return contact_strength(tac_t, tac_0, self.content_box, self.pool, self.top_q)

    @torch.no_grad()
    def allocate(self, s: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(n, active), both (B, V) on the CPU: per-view token counts and the active flags."""
        # Unset thresholds would make every view inactive: a rule model silently running uniform10.
        problem = self.missing_statistics()
        if problem:
            raise RuntimeError(problem)
        return allocate(
            s, self.threshold, self.mode, valid, sibling=self.sibling, na_hist=self.na_hist, return_active=True
        )

    def embed(self, tok: torch.Tensor, view: int, n_tokens: int) -> torch.Tensor:
        """Add the view and level embeddings to (b, n_tokens, D) projected tokens of one view."""
        return tok + self.view_embed[view].to(tok.dtype) + self.level_embed[LEVEL_INDEX[n_tokens]].to(tok.dtype)


def _check_loaded_statistics(
    module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
) -> None:
    """``load_state_dict`` pre-hook: the checkpoint's statistics win, but never silently.

    A checkpoint without ``input_res.*`` (B0, released) leaves the configured values in place. A 1A
    checkpoint must have been trained in this mode (``mode_id``) and restores the statistics it was
    trained with; if they differ from the configured ones (another stats JSON), that is logged. A
    checkpoint whose thresholds were never set (a uniform run) cannot drive a threshold mode and is
    rejected.
    """
    del local_metadata, strict, missing_keys, unexpected_keys
    incoming_mode = state_dict.get(prefix + "mode_id")
    if incoming_mode is not None and incoming_mode.numel() == 1 and int(incoming_mode) != int(module.mode_id):
        trained = int(incoming_mode)
        trained = ALLOCATION_MODES[trained] if 0 <= trained < len(ALLOCATION_MODES) else trained
        error_msgs.append(
            f"{prefix}mode_id: the checkpoint was trained with input_res_mode={trained!r} but the config "
            f"uses {module.mode!r}"
        )
    configured = bool(torch.isfinite(module.tau).all()) or float(module.na_hist.sum()) > 0
    for name in ("tau", "tau_floor", "na_hist"):
        incoming = state_dict.get(prefix + name)
        if incoming is None:
            continue
        current = getattr(module, name)
        if incoming.shape != current.shape:
            continue  # load_state_dict reports the size mismatch itself
        if name == "tau" and module.mode in THRESHOLD_MODES and not bool(torch.isfinite(incoming).all()):
            error_msgs.append(
                f"{prefix}tau in the checkpoint is {incoming.tolist()} (never set); "
                f"mode {module.mode!r} needs thresholds"
            )
        elif name == "na_hist" and module.mode in NA_HIST_MODES and float(incoming.sum()) <= 0:
            error_msgs.append(f"{prefix}na_hist in the checkpoint is empty; mode 'random' needs P(n_a)")
        elif not torch.equal(incoming.to(current.dtype).cpu(), current.detach().cpu()):
            if configured:
                logging.warning(
                    "%s%s: using the checkpoint's %s instead of the configured %s",
                    prefix, name, incoming.tolist(), current.tolist(),
                )
            else:
                logging.info("%s%s: %s from the checkpoint (none configured)", prefix, name, incoming.tolist())
