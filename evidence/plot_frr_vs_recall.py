#!/usr/bin/env python3
"""plot_frr_vs_recall.py: draw the model card's figure, frr_vs_recall.png, from this bundle's scorecards.

Harmless OR-Bench-Hard prompts wrongly blocked (false-refusal rate, 1,319 prompts) against harmful OR-Bench-Toxic
prompts caught (recall, 655 prompts). Item labels for OR-Bench are carried in the scorecards, so this runs on the
bundle as shipped (attach_labels.py is not needed).

    pip install numpy matplotlib
    python3 plot_frr_vs_recall.py <bundle folder> [--out frr_vs_recall.png]

Curves (release 2.0's published weights, omni-moderation snapshot omni-moderation-2024-09-26, and GPT-4o
gpt-4o-2024-11-20 with the project's guard prompt): for each k = 1..655 the threshold is the k-th highest harmful
score; recall is the share of harmful prompts scoring at or above it (with tied scores, that can be more than k)
and the false-refusal rate the share of harmless prompts at or above it. omni's score is the highest of its 13
category scores; GPT-4o's is P(block), its probability of answering BLOCK.
Bands: 95% bootstrap intervals for each curve's false-refusal rate at each of its thresholds, resampling the 1,319
harmless prompts with the thresholds fixed. The draws are reproduce_card.py's OR-Bench harmless-prompt draws
(numpy default_rng([13, zlib.crc32("orbench")]), the first 10,000 x 1,319 integers), the same for every curve, so
a band read at a table threshold equals that table's interval.
Points: release 2.0's three thresholds on its curve (0.90, the default 0.455 and the pre-screen 0.043, each
labelled with its values); omni at its own `flagged` decision; GPT-4o at its own decision (P(block) >= 0.5); and
release 2.0 at 0.455 with GPT-4o reviewing only what it flags (a prompt is blocked only if both flag it). The
legend gives the comparison points' values.
"""
import argparse
import json
import pathlib
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PUB = "classifier__v6_R3_s13"                               # release 2.0, published weights
OMNI = "classifier__openai_omni-moderation-2024-09-26"
GPT4O = "guard__gpt-4o-2024-11-20"
T_BAL = 0.45514049134720697                                 # release 2.0's default (balanced) threshold
T_SCREEN = 0.04342702474724448                              # release 2.0's pre-screen threshold
B = 10000                                                   # bootstrap draws for the bands

# Palette: categorical slots 1-4 of the reference palette, validated in light mode on #fcfcfb (all checks pass;
# aqua and yellow are below 3:1 contrast, so every point has its own marker shape, an outline and a legend entry
# that gives its values).
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_V2, C_OMNI, C_G, C_PAIR = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def load(sc, model, split):
    rows = json.loads((sc / f"{model}__{split}.json").read_text())["rows"]
    return np.array([r["score"] for r in rows], float), np.array([int(r["decision"]) for r in rows], int)


def curve(benign, unsafe):
    order = np.sort(unsafe)[::-1]
    thr = order[np.arange(len(unsafe))]
    rec = np.array([(unsafe >= t).mean() for t in thr])
    frr = np.array([(benign >= t).mean() for t in thr])
    return rec, frr, thr


def band(benign, thr, I, chunk=1000):
    """95% interval of the false-refusal rate at each fixed threshold, over the harmless-prompt draws I."""
    n = len(benign)
    o = np.argsort(benign, kind="mergesort")
    pos = np.searchsorted(benign[o], thr, "left")          # harmless prompts below each threshold
    draws = []
    for c in range(0, I.shape[0], chunk):
        idx = I[c:c + chunk]
        m = idx.shape[0]
        cnt = np.bincount((idx + (np.arange(m) * n)[:, None]).ravel(), minlength=m * n).reshape(m, n)
        cum = np.concatenate([np.zeros((m, 1)), np.cumsum(cnt[:, o], 1)], 1)
        draws.append((n - cum[:, pos]) / n)
    d = np.concatenate(draws)
    return np.percentile(d, 2.5, axis=0), np.percentile(d, 97.5, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--out", default="frr_vs_recall.png")
    a = ap.parse_args()
    sc = pathlib.Path(a.bundle) / "scorecards" / "test"

    vb, _ = load(sc, PUB, "orbench_hard")
    vu, _ = load(sc, PUB, "orbench_toxic")
    ob, odb = load(sc, OMNI, "orbench_hard")
    ou, odu = load(sc, OMNI, "orbench_toxic")
    gsb, gb = load(sc, GPT4O, "orbench_hard")
    gsu, gu = load(sc, GPT4O, "orbench_toxic")
    assert ((gsb >= 0.5) == (gb == 1)).all() and ((gsu >= 0.5) == (gu == 1)).all()   # decision = P(block) >= 0.5

    rv, fv, tv = curve(vb, vu)
    ro, fo, to = curve(ob, ou)
    rg, fg, tg = curve(gsb, gsu)
    # reproduce_card.py's OR-Bench stream: its first draws are the harmless prompts (B x 1,319), the same for every model
    I = np.random.default_rng([13, zlib.crc32(b"orbench")]).integers(0, len(vb), (B, len(vb)))
    bands = [band(vb, tv, I), band(ob, to, I), band(gsb, tg, I)]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)

    for (lo, hi), r, c in zip(bands, (rv, ro, rg), (C_V2, C_OMNI, C_G)):
        ax.fill_between(100 * r, 100 * lo, 100 * hi, color=c, alpha=0.16, linewidth=0, step="post")
    ax.plot(100 * rv, 100 * fv, color=C_V2, linewidth=2, label="release 2.0, published weights (curve, 95% band)")
    ax.plot(100 * ro, 100 * fo, color=C_OMNI, linewidth=2, label="omni-moderation-2024-09-26 (curve, 95% band)")
    ax.plot(100 * rg, 100 * fg, color=C_G, linewidth=2, linestyle=(0, (5, 2)),
            label="GPT-4o as a guard, score P(block) (curve, 95% band)")

    def point(r, f, marker, color, size, text, edge=SURFACE):
        ax.plot(100 * r, 100 * f, marker, markersize=size, color=color, markeredgecolor=edge, markeredgewidth=1.5,
                zorder=5, linestyle="none", label=f"{text}\n{100 * r:.1f}% caught, {100 * f:.1f}% wrongly blocked")

    # release 2.0's three thresholds, on its curve: label offsets (points) are placed by hand to clear the curves,
    # the bands and the legend
    for t, name, filled, xy, ha, sep in [(0.90, "0.90 (block)", False, (6, 42), "center", ", "),
                                         (T_BAL, "0.455 (default)", True, (-12, 34), "right", ", "),
                                         (T_SCREEN, "0.043 (pre-screen)", False, (-22, -104), "right", ",\n")]:
        r, f = (vu >= t).mean(), (vb >= t).mean()
        ax.plot(100 * r, 100 * f, "o", markersize=8.5, color=C_V2 if filled else SURFACE, markeredgecolor=C_V2,
                markeredgewidth=2, zorder=6, linestyle="none")
        rel = {"center": (0.5, 0.0), "right": (1.0, 0.5)}[ha]
        ax.annotate(f"{name}\n{100 * r:.1f}% caught{sep}{100 * f:.1f}% blocked", (100 * r, 100 * f), xytext=xy,
                    textcoords="offset points", fontsize=8.4, color=INK, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8, shrinkA=2, shrinkB=5, relpos=rel))
    point(odu.mean(), odb.mean(), "s", C_OMNI, 8, "omni at its flagged decision")
    point(gu.mean(), gb.mean(), "D", C_G, 8, "GPT-4o at its own decision", edge=INK2)
    pu_, pb_ = (vu >= T_BAL) & (gu == 1), (vb >= T_BAL) & (gb == 1)
    point(pu_.mean(), pb_.mean(), "*", C_PAIR, 15, "release 2.0 at 0.455, GPT-4o reviewing its flags", edge=INK2)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Harmful prompts caught: recall on OR-Bench-Toxic (655 prompts), %")
    ax.set_ylabel("Harmless prompts wrongly blocked:\nfalse-refusal rate on OR-Bench-Hard (1,319 prompts), %")
    ax.set_title("Wrongly blocked vs caught on OR-Bench (lower is better)", loc="left", fontsize=12, color=INK)
    leg = ax.legend(loc="upper left", frameon=True, framealpha=1, facecolor=SURFACE, edgecolor=GRID,
                    fontsize=8.6, labelspacing=0.55, borderpad=0.6)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(a.out, facecolor=SURFACE)
    above = int((fv > fo).sum())
    g_below = int((fg < fv).sum())
    lo, hi = band(vb, np.array([T_BAL]), I)                 # equals the card's OR-Bench interval at 0.455
    print(f"harmless prompts wrongly blocked at 0.455: {100 * (vb >= T_BAL).mean():.1f}% [{100 * lo[0]:.1f}, {100 * hi[0]:.1f}]")
    print(f"wrote {a.out}; matched-recall points where release 2.0 blocks more harmless prompts than omni: {above}; "
          f"where GPT-4o blocks fewer than release 2.0: {g_below} of {len(fv)}")


if __name__ == "__main__":
    main()
