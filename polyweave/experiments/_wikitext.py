"""WikiText-2 corpus helper for the GPT-2 MLP-distillation experiment.

Returns the raw text of a WikiText-2 split, caching it to a local ``.txt`` so
repeat runs (and the generic ``cfg.text_paths`` path) need neither the optional
``datasets`` dependency nor a network round-trip after the first fetch.

WikiText-2 (Merity et al., 2016) is the standard small language-modelling
benchmark: ~2M training tokens of cleaned Wikipedia, reproducible, and tiny
enough for a 6 GB GPU. We use the *raw* (``wikitext-2-raw-v1``) variant so the
tokenizer sees real text (the non-raw variant is pre-tokenised with ``<unk>``).

Note GPT-2 was trained on WebText, not Wikipedia, so absolute perplexities here
are higher than GPT-2's headline numbers — but we only ever read *deltas* (PPL
with the distilled layer swapped in vs the original block), for which a fixed,
reproducible corpus is exactly what we want.
"""

from __future__ import annotations

from pathlib import Path

_HF_CONFIG = "wikitext-2-raw-v1"
# HF exposes train/validation/test; accept a couple of friendly aliases.
_SPLIT_ALIASES = {"val": "validation", "valid": "validation", "dev": "validation"}


def wikitext2_text(split: str = "train", cache_dir: str = "data") -> str:
    """Raw text of one WikiText-2 split (``train`` / ``validation`` / ``test``).

    Caches to ``{cache_dir}/wikitext2_{split}.txt`` on first use. On a cache miss
    the text is pulled via the optional ``datasets`` package; if that is not
    installed and no cache exists, a clear ``ImportError`` tells the caller how to
    proceed (``pip install datasets`` or pre-populate the cache file).
    """
    split = _SPLIT_ALIASES.get(split, split)
    cache = Path(cache_dir) / f"wikitext2_{split}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    try:
        from datasets import load_dataset  # optional dep, lazy import
    except ImportError as exc:  # pragma: no cover - exercised only without datasets
        raise ImportError(
            "Loading WikiText-2 needs the optional 'datasets' package "
            "(`pip install datasets`), or a pre-downloaded cache file at "
            f"{cache}. Install datasets or drop the raw text there."
        ) from exc

    ds = load_dataset("wikitext", _HF_CONFIG, split=split)
    text = "\n".join(ds["text"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text
