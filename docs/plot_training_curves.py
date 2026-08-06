"""
Render the training curves in training_history.csv to SVG.

    python docs/plot_training_curves.py

Two charts, deliberately drawn differently:

  Recall is ONE line across all 80 epochs. Recall@IoU0.5 is a weight-independent
  count of matched ground-truth boxes, so it means the same thing before and
  after the config change and the two halves are directly comparable.

  Val loss is TWO DISCONNECTED segments. The config change bumped
  LOSS_SEMANTICS_VERSION 4 -> 5: num_queries went 200 -> 100 (halving the
  background mass in the query-mean classification term), the aux loss went
  0 -> 0.4 and CIoU 2.0 -> 5.0. The number on the y axis is a different
  quantity either side of epoch 50, so joining them would draw a cliff that
  represents a redefinition rather than a regression.

Colours are chosen to read on both light and dark backgrounds, and the
background is transparent, so one file serves both GitHub themes.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "training_history.csv")

COLOR_A = "#2f81f7"   # config A - readable on white and on #0d1117
COLOR_B = "#db6d28"   # config B
CHROME = "#7d8590"    # axes, ticks, labels


def load():
    with open(CSV_PATH, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def style(ax, xlabel, ylabel, title):
    ax.set_title(title, color=CHROME, fontsize=12, pad=12)
    ax.set_xlabel(xlabel, color=CHROME, fontsize=10)
    ax.set_ylabel(ylabel, color=CHROME, fontsize=10)
    ax.tick_params(colors=CHROME, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(CHROME)
        ax.spines[side].set_alpha(0.4)
    ax.grid(True, color=CHROME, alpha=0.15, linewidth=0.7)
    ax.set_axisbelow(True)


def boundary(ax, x=50.5):
    ax.axvline(x, color=CHROME, alpha=0.5, linestyle="--", linewidth=1)


def save(fig, name):
    out = os.path.join(HERE, name)
    fig.savefig(out, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    rows = load()
    a = [r for r in rows if r["config"] == "A"]
    b = [r for r in rows if r["config"] == "B"]

    ep = lambda rs: [int(r["epoch"]) for r in rs]
    col = lambda rs, k: [float(r[k]) for r in rs]

    # ---- Validation loss -------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(ep(a), col(a, "val_loss"), color=COLOR_A, linewidth=1.8,
            label="Config A (epochs 1-50)")
    ax.plot(ep(b), col(b, "val_loss"), color=COLOR_B, linewidth=1.8,
            label="Config B (epochs 51-80)")
    boundary(ax)
    style(ax, "Epoch", "Validation loss",
          "Validation loss per epoch  -  segments are NOT comparable")
    ax.annotate("loss redefined here\n(LOSS_SEMANTICS_VERSION 4 to 5)",
                xy=(50.5, max(col(b, "val_loss"))), xytext=(28, 5.6),
                color=CHROME, fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=CHROME, alpha=0.6))
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(CHROME)
    save(fig, "val_loss_curve.svg")

    # ---- Recall ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    # Joined across the boundary on purpose: this metric IS comparable, and the
    # dip at 51 is a real re-adaptation cost, not an artefact of rescaling.
    ax.plot(ep(a), col(a, "recall_iou05"), color=COLOR_A, linewidth=1.8,
            label="Config A (epochs 1-50)")
    ax.plot(ep(a)[-1:] + ep(b), col(a, "recall_iou05")[-1:] + col(b, "recall_iou05"),
            color=COLOR_B, linewidth=1.8, label="Config B (epochs 51-80)")
    boundary(ax)
    style(ax, "Epoch", "Recall@IoU0.5 (%)",
          "Validation Recall@IoU0.5 per epoch")
    ax.annotate("config change:\naspect buckets, crop aug,\n200 to 100 queries",
                xy=(51, 75.16), xytext=(56, 66.5),
                color=CHROME, fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=CHROME, alpha=0.6))
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(CHROME)
    save(fig, "recall_curve.svg")


if __name__ == "__main__":
    main()
