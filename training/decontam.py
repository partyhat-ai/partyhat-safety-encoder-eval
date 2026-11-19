#!/usr/bin/env python
"""
decontam.py — text normalisation and duplicate / near-duplicate detection against test sets.

Purpose
-------
Before ANY external / synthetic / proprietary row is allowed near training, it
must be proven NOT to be a duplicate or near-duplicate of a benchmark TEST item.
The benchmark here is the ToxicChat test split (loaded through the locked
tc_common code path, never re-derived).

Why this matters (the LMSYS trap)
---------------------------------
ToxicChat is itself a curated/annotated subsample of LMSYS-Chat-1M. Any external
corpus drawn from LMSYS-Chat-1M (a very common move) therefore almost certainly
contains ToxicChat *test* prompts verbatim, or lightly paraphrased. If those
leak into training, test F1/AUPRC are inflated and the comparison to the
published T5 bar is no longer honest.

Three layers of defense, cheapest first:
  1. EXACT pass        — normalized-text hash equality. Catches verbatim copies
                         and trivial case/whitespace/punctuation reformats.
  2. NEAR-DUP pass     — exact Jaccard over word-shingle sets, accelerated by an
                         inverted index (shingle -> reference ids) so we only
                         compare candidates against references that actually
                         share a shingle. Catches small edits / reorderings.
  3. EMBEDDING pass    — OPTIONAL semantic paraphrase detection via
     (the LMSYS trap)    sentence-transformers cosine similarity. This is the
                         only layer that catches a genuine *paraphrase* with no
                         lexical overlap, which is exactly the LMSYS failure
                         mode. REQUIRED before trusting any LMSYS-derived data.

Design constraints (match tc_common.py / train.py house style)
--------------------------------------------------------------
  * Importable with ONLY numpy + pandas + stdlib. sentence-transformers and
    datasketch are STRICTLY optional and lazily imported; their absence never
    crashes the default path — at worst it skips the embedding layer with a
    clear stderr warning.
  * FP32-careful: embeddings are compared in float32 with explicit L2
    normalization so cosine is a plain dot product (no float64 drift, no
    surprise upcasts).
  * The pure-python inverted-index Jaccard is the default and is EXACT (not a
    MinHash estimate). It is fast at our scale (references ~5k, candidates up to
    ~100k). datasketch MinHash-LSH is offered only as an optional accelerator
    behind a flag, and only if the library is actually installed.

Public API
----------
  decontaminate(candidates, references, jaccard=0.8, shingle=4, embedding=False)
      -> dict(kept_idx, removed_idx, matches)
  decontaminate_df(df, text_field, references, **kw)
      -> (clean_df, report_dict)
  clean_against_toxicchat_test(df, text_field, **kw)
      -> (clean_df, report)   # references = ToxicChat test via tc_common

CLI
---
  python decontam.py --candidates data.csv --text-field user_input \\
        --against test [--jaccard 0.8] [--shingle 4] [--embedding] --out clean.csv

  --against accepts 'test' (ToxicChat test, via tc_common.load_test) or a path
  to a references CSV (then pair it with --ref-field).
"""
import argparse
import json
import re
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Defaults — single source of truth, mirrored into the CLI argparse below.     #
# --------------------------------------------------------------------------- #
DEFAULT_JACCARD = 0.8       # near-dup Jaccard threshold (>= flags)
DEFAULT_SHINGLE = 4         # word-shingle width, in words
DEFAULT_EMB_THRESHOLD = 0.9  # cosine sim above which a candidate is a paraphrase
DEFAULT_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------------------------------- #
# 1. Text normalization.                                                       #
# --------------------------------------------------------------------------- #
# Deliberately simple and documented so the behavior is obvious and stable:
#   * lowercase
#   * map every run of non-alphanumeric characters to a single space
#     (this folds away punctuation-only differences AND collapses whitespace)
#   * strip leading/trailing space
# We keep digits because numeric content (e.g. "delete user 4821") is meaningful
# for these prompts. The result is a clean, space-joined token stream that both
# the exact-hash and the shingle passes consume.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_text(text) -> str:
    """Lowercase, fold all punctuation/whitespace to single spaces, strip.

    NaN / None / non-str inputs normalize to the empty string so that empty or
    malformed candidates are handled gracefully (they simply never match).
    """
    if text is None:
        return ""
    # pandas can hand us float NaN for missing cells.
    if isinstance(text, float) and np.isnan(text):
        return ""
    s = str(text).lower()
    s = _NON_ALNUM.sub(" ", s)
    return s.strip()


def shingles(normalized: str, w: int = DEFAULT_SHINGLE) -> set:
    """Word-shingle set for an already-normalized string.

    A shingle is a window of `w` consecutive words, joined by single spaces.
    For documents shorter than `w` words we fall back to the single shingle
    consisting of the whole (normalized) doc, so short prompts are still
    comparable rather than silently producing an empty set.
    """
    words = normalized.split()
    if not words:
        return set()
    if len(words) < w:
        return {" ".join(words)}
    return {" ".join(words[i:i + w]) for i in range(len(words) - w + 1)}


def _jaccard(a: set, b: set) -> float:
    """Exact Jaccard similarity |A∩B| / |A∪B|. Empty-vs-empty -> 0.0."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a) + len(b) - inter
    return inter / union


# --------------------------------------------------------------------------- #
# 2. The core engine.                                                          #
# --------------------------------------------------------------------------- #
def decontaminate(
    candidates,
    references,
    jaccard: float = DEFAULT_JACCARD,
    shingle: int = DEFAULT_SHINGLE,
    embedding: bool = False,
    emb_threshold: float = DEFAULT_EMB_THRESHOLD,
    emb_model: str = DEFAULT_EMB_MODEL,
    use_lsh: bool = False,
):
    """Flag candidates that duplicate / near-duplicate any reference.

    Parameters
    ----------
    candidates : list[str]   the rows under suspicion (e.g. external training data)
    references : list[str]   the protected set (e.g. ToxicChat TEST prompts)
    jaccard    : float       near-dup threshold; Jaccard >= this flags (default 0.8)
    shingle    : int         word-shingle width (default 4)
    embedding  : bool        run the optional semantic paraphrase pass
    emb_threshold : float    cosine sim above which a candidate is a paraphrase
    emb_model  : str         sentence-transformers model id for the embedding pass
    use_lsh    : bool        if datasketch is installed, use MinHash-LSH to gather
                             near-dup candidate pairs instead of the inverted index
                             (pure-python inverted index is the default)

    Returns
    -------
    dict with keys:
      kept_idx    : sorted list of candidate indices that survived (clean)
      removed_idx : sorted list of candidate indices flagged as contaminated
      matches     : list of dicts {cand_idx, ref_idx, kind, score} where kind is
                    one of 'exact' | 'nahash' | 'embedding'. A candidate may have
                    multiple matches; it is removed if it has >= 1 match of any
                    kind. (`nahash` == near-dup-by-hash-index / shingle Jaccard.)

    Determinism: every output list is sorted; ties in the inverted-index walk are
    resolved by ascending reference index. Re-running on the same inputs yields
    byte-identical output.
    """
    candidates = list(candidates)
    references = list(references)
    n_cand = len(candidates)

    # Normalize once. Keep the normalized strings and the shingle sets around;
    # both passes reuse them so we never re-tokenize.
    cand_norm = [normalize_text(c) for c in candidates]
    ref_norm = [normalize_text(r) for r in references]
    cand_sh = [shingles(s, shingle) for s in cand_norm]
    ref_sh = [shingles(s, shingle) for s in ref_norm]

    matches = []          # accumulates every (cand, ref, kind, score)
    removed = set()       # candidate indices flagged by ANY pass

    # ---- EXACT pass ------------------------------------------------------- #
    # Map each non-empty normalized reference text to the *smallest* ref index
    # that produced it (deterministic representative). A candidate whose
    # normalized text hits the map is an exact (post-normalization) duplicate.
    exact_index = {}
    for ri, rn in enumerate(ref_norm):
        if rn and rn not in exact_index:
            exact_index[rn] = ri
    for ci, cn in enumerate(cand_norm):
        if not cn:
            continue
        ri = exact_index.get(cn)
        if ri is not None:
            matches.append(
                {"cand_idx": ci, "ref_idx": ri, "kind": "exact", "score": 1.0}
            )
            removed.add(ci)

    # ---- NEAR-DUP pass (shingle Jaccard) ---------------------------------- #
    # Build inverted index shingle -> sorted list of reference ids. For each
    # candidate we union the posting lists of its shingles to get the ONLY
    # references that can possibly clear the Jaccard bar (a pair with zero shared
    # shingles has Jaccard 0). We then compute exact Jaccard against that short
    # list. This is exact, not an estimate.
    if use_lsh:
        lsh_matches = _near_dup_lsh(
            cand_sh, ref_sh, jaccard, shingle, removed
        )
        matches.extend(lsh_matches)
        for m in lsh_matches:
            removed.add(m["cand_idx"])
    else:
        inverted = {}
        for ri, sh in enumerate(ref_sh):
            for g in sh:
                inverted.setdefault(g, []).append(ri)
        # posting lists are already in ascending ref order (we iterate ri up).

        for ci, csh in enumerate(cand_sh):
            if ci in removed or not csh:
                # already an exact dup, or nothing to compare — skip the work.
                continue
            cand_refs = set()
            for g in csh:
                post = inverted.get(g)
                if post:
                    cand_refs.update(post)
            if not cand_refs:
                continue
            best_ri, best_score = -1, 0.0
            # iterate refs in ascending order for deterministic tie-breaking.
            for ri in sorted(cand_refs):
                s = _jaccard(csh, ref_sh[ri])
                if s > best_score or (s == best_score and best_ri == -1):
                    best_score, best_ri = s, ri
            if best_score >= jaccard:
                matches.append(
                    {
                        "cand_idx": ci,
                        "ref_idx": best_ri,
                        "kind": "nahash",
                        "score": float(best_score),
                    }
                )
                removed.add(ci)

    # ---- EMBEDDING pass (optional, the LMSYS trap) ------------------------ #
    if embedding:
        emb_matches = _embedding_pass(
            candidates, references, emb_threshold, emb_model, skip=removed
        )
        if emb_matches is not None:
            matches.extend(emb_matches)
            for m in emb_matches:
                removed.add(m["cand_idx"])

    kept = sorted(set(range(n_cand)) - removed)
    # Sort matches deterministically: by candidate, then ref, then kind.
    matches.sort(key=lambda m: (m["cand_idx"], m["ref_idx"], m["kind"]))
    return {
        "kept_idx": kept,
        "removed_idx": sorted(removed),
        "matches": matches,
    }


# --------------------------------------------------------------------------- #
# Optional accelerators / passes — lazily imported so the defaults never crash. #
# --------------------------------------------------------------------------- #
def _near_dup_lsh(cand_sh, ref_sh, jaccard, shingle, already_removed):
    """Optional datasketch MinHash-LSH path for the near-dup pass.

    Used only when --use-lsh is passed AND datasketch is installed. We use LSH
    purely to GATHER candidate->reference pairs cheaply, then confirm each pair
    with the same EXACT Jaccard over the true shingle sets, so the flagging
    decision is identical to the inverted-index path (LSH only changes how we
    find pairs, never the threshold semantics).
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print(
            "[decontam] --use-lsh requested but datasketch is not installed; "
            "falling back to exact inverted-index Jaccard.",
            file=sys.stderr,
        )
        # Re-run the default inverted-index logic by signalling the caller.
        # Simplest: replicate the inverted-index here so behavior is unchanged.
        inverted = {}
        for ri, sh in enumerate(ref_sh):
            for g in sh:
                inverted.setdefault(g, []).append(ri)
        out = []
        for ci, csh in enumerate(cand_sh):
            if ci in already_removed or not csh:
                continue
            cand_refs = set()
            for g in csh:
                post = inverted.get(g)
                if post:
                    cand_refs.update(post)
            best_ri, best_score = -1, 0.0
            for ri in sorted(cand_refs):
                s = _jaccard(csh, ref_sh[ri])
                if s > best_score:
                    best_score, best_ri = s, ri
            if best_score >= jaccard:
                out.append({"cand_idx": ci, "ref_idx": best_ri,
                            "kind": "nahash", "score": float(best_score)})
        return out

    num_perm = 128

    def _mh(sh):
        m = MinHash(num_perm=num_perm)
        for g in sh:
            m.update(g.encode("utf-8"))
        return m

    lsh = MinHashLSH(threshold=jaccard, num_perm=num_perm)
    ref_mh = []
    for ri, sh in enumerate(ref_sh):
        m = _mh(sh) if sh else None
        ref_mh.append(m)
        if m is not None:
            lsh.insert(str(ri), m)

    out = []
    for ci, csh in enumerate(cand_sh):
        if ci in already_removed or not csh:
            continue
        cand_refs = [int(x) for x in lsh.query(_mh(csh))]
        best_ri, best_score = -1, 0.0
        for ri in sorted(cand_refs):
            s = _jaccard(csh, ref_sh[ri])   # exact confirmation
            if s > best_score:
                best_score, best_ri = s, ri
        if best_score >= jaccard:
            out.append({"cand_idx": ci, "ref_idx": best_ri,
                        "kind": "nahash", "score": float(best_score)})
    return out


def _embedding_pass(candidates, references, threshold, model_id, skip):
    """Optional sentence-transformers semantic paraphrase pass.

    Returns a list of match dicts, or None if sentence-transformers is not
    installed (in which case we warn loudly on stderr and skip — never crash).

    Only candidates NOT already flagged (`skip`) are embedded/compared, which
    keeps this — the slow pass — as cheap as possible. FP32-careful: embeddings
    are cast to float32 and L2-normalized so cosine similarity is a plain matmul
    with no float64 round-trips.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "[decontam] WARNING: embedding pass requested but "
            "sentence-transformers is NOT installed. Skipping the semantic "
            "paraphrase layer.\n"
            "           This layer is REQUIRED before trusting any "
            "LMSYS-derived data (verbatim/lexical dedup misses paraphrases).\n"
            "           Install with: pip install sentence-transformers",
            file=sys.stderr,
        )
        return None

    model = SentenceTransformer(model_id)

    cand_idx = [i for i in range(len(candidates)) if i not in skip]
    if not cand_idx:
        return []

    def _embed(texts):
        e = model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False,
            normalize_embeddings=False, batch_size=64,
        )
        e = np.asarray(e, dtype=np.float32)
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0           # guard div-by-zero on empty text
        return e / norms                    # L2-normalized -> cosine == dot

    ref_emb = _embed([str(r) for r in references])              # [R, D] f32
    cand_emb = _embed([str(candidates[i]) for i in cand_idx])   # [C, D] f32

    # Cosine sim matrix in chunks to bound memory: [C_chunk, R].
    out = []
    chunk = 512
    for start in range(0, cand_emb.shape[0], chunk):
        block = cand_emb[start:start + chunk]                   # [b, D]
        sims = block @ ref_emb.T                                # [b, R] f32
        best_ri = np.argmax(sims, axis=1)
        best_sc = sims[np.arange(sims.shape[0]), best_ri]
        for j in range(block.shape[0]):
            if best_sc[j] >= threshold:
                ci = cand_idx[start + j]
                out.append({
                    "cand_idx": int(ci),
                    "ref_idx": int(best_ri[j]),
                    "kind": "embedding",
                    "score": float(best_sc[j]),
                })
    return out


# --------------------------------------------------------------------------- #
# 3. DataFrame conveniences.                                                   #
# --------------------------------------------------------------------------- #
def decontaminate_df(df: pd.DataFrame, text_field: str, references, **kw):
    """Decontaminate a DataFrame's `text_field` against `references`.

    Returns (clean_df, report). `clean_df` is the rows that survived, with the
    original index reset. `report` is a JSON-serializable summary plus the raw
    match list (with cand_idx aligned to df's *positional* index 0..n-1).
    """
    if text_field not in df.columns:
        raise KeyError(
            f"text_field {text_field!r} not in df columns {list(df.columns)}"
        )
    candidates = df[text_field].tolist()
    result = decontaminate(candidates, references, **kw)

    clean_df = df.iloc[result["kept_idx"]].reset_index(drop=True)

    by_kind = {}
    removed_set = set(result["removed_idx"])
    # Count each removed candidate ONCE, by the strongest pass that caught it.
    # Priority: exact > nahash > embedding (cheapest/strongest evidence first).
    kind_for_cand = {}
    priority = {"exact": 0, "nahash": 1, "embedding": 2}
    for m in result["matches"]:
        ci = m["cand_idx"]
        if ci not in kind_for_cand or priority[m["kind"]] < priority[kind_for_cand[ci]]:
            kind_for_cand[ci] = m["kind"]
    for ci in removed_set:
        k = kind_for_cand.get(ci, "nahash")
        by_kind[k] = by_kind.get(k, 0) + 1

    report = {
        "n_candidates": len(candidates),
        "n_references": len(references),
        "n_removed": len(result["removed_idx"]),
        "n_kept": len(result["kept_idx"]),
        "removed_frac": (len(result["removed_idx"]) / len(candidates))
        if candidates else 0.0,
        "breakdown_by_kind": by_kind,
        "params": {k: v for k, v in kw.items()},
        "example_matches": result["matches"][:10],
        "matches": result["matches"],
        "removed_idx": result["removed_idx"],
        "kept_idx": result["kept_idx"],
    }
    return clean_df, report


def clean_against_toxicchat_test(df: pd.DataFrame, text_field: str, **kw):
    """Decontaminate `df[text_field]` against the ToxicChat TEST prompts.

    References come from the LOCKED tc_common code path
    (load_test()[INPUT_FIELD]) so they are exactly the ToxicChat test items. Set TOXICCHAT_DIR to load the local CSVs (otherwise tc_common falls back
    to the gated Hub download). Returns (clean_df, report).
    """
    from tc_common import INPUT_FIELD, load_test

    references = load_test()[INPUT_FIELD].tolist()
    return decontaminate_df(df, text_field, references, **kw)


# --------------------------------------------------------------------------- #
# CLI.                                                                         #
# --------------------------------------------------------------------------- #
def _build_argparser():
    p = argparse.ArgumentParser(
        description=(
            "STEP 0 contamination/dedup gate: remove candidate rows that "
            "duplicate or near-duplicate a protected reference set (default: "
            "the ToxicChat test split)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--candidates", required=True,
                   help="CSV of candidate rows to clean.")
    p.add_argument("--text-field", required=True,
                   help="Column in --candidates holding the text to dedup on.")
    p.add_argument("--against", default="test",
                   help="'test' for ToxicChat test (via tc_common), or a path "
                        "to a references CSV (pair with --ref-field).")
    p.add_argument("--ref-field", default=None,
                   help="Text column in the references CSV (when --against is a "
                        "path). Defaults to --text-field if omitted.")
    p.add_argument("--jaccard", type=float, default=DEFAULT_JACCARD,
                   help="Near-dup Jaccard threshold (>= flags).")
    p.add_argument("--shingle", type=int, default=DEFAULT_SHINGLE,
                   help="Word-shingle width.")
    p.add_argument("--embedding", action="store_true",
                   help="Run the optional semantic paraphrase pass "
                        "(needs sentence-transformers; REQUIRED for LMSYS data).")
    p.add_argument("--emb-threshold", type=float, default=DEFAULT_EMB_THRESHOLD,
                   help="Cosine-sim threshold for the embedding pass.")
    p.add_argument("--emb-model", default=DEFAULT_EMB_MODEL,
                   help="sentence-transformers model id for the embedding pass.")
    p.add_argument("--use-lsh", action="store_true",
                   help="Use datasketch MinHash-LSH to gather near-dup pairs "
                        "(optional accelerator; exact inverted index is default).")
    p.add_argument("--out", required=True,
                   help="Path to write the cleaned CSV.")
    return p


def main(argv=None):
    args = _build_argparser().parse_args(argv)

    cand_df = pd.read_csv(args.candidates)
    if args.text_field not in cand_df.columns:
        raise SystemExit(
            f"--text-field {args.text_field!r} not in candidates columns "
            f"{list(cand_df.columns)}"
        )

    kw = dict(
        jaccard=args.jaccard,
        shingle=args.shingle,
        embedding=args.embedding,
        emb_threshold=args.emb_threshold,
        emb_model=args.emb_model,
        use_lsh=args.use_lsh,
    )

    if args.against == "test":
        clean_df, report = clean_against_toxicchat_test(
            cand_df, args.text_field, **kw
        )
        report["references"] = "toxicchat-test"
    else:
        ref_field = args.ref_field or args.text_field
        ref_df = pd.read_csv(args.against)
        if ref_field not in ref_df.columns:
            raise SystemExit(
                f"--ref-field {ref_field!r} not in references columns "
                f"{list(ref_df.columns)}"
            )
        references = ref_df[ref_field].tolist()
        clean_df, report = decontaminate_df(
            cand_df, args.text_field, references, **kw
        )
        report["references"] = args.against

    clean_df.to_csv(args.out, index=False)
    report["out"] = args.out

    # Print a compact JSON report (drop the bulky full match/idx lists; keep the
    # headline numbers, the per-kind breakdown, and a few example matches).
    summary = {
        "references": report["references"],
        "out": report["out"],
        "n_candidates": report["n_candidates"],
        "n_references": report["n_references"],
        "n_removed": report["n_removed"],
        "n_kept": report["n_kept"],
        "removed_frac": round(report["removed_frac"], 6),
        "breakdown_by_kind": report["breakdown_by_kind"],
        "params": report["params"],
        "example_matches": report["example_matches"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
