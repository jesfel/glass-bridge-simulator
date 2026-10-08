"""Article figures from the saved sweep results (results/*.csv). Run:

    python -m glass_bridge.figures            # writes figures/*.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import exact

# Reference palette (light mode; Medium pages are white)
SURFACE, INK, INK_2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, RED, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#f0efec"
BLUE_RAMP = {0.0: "#86b6ef", 0.5: "#2a78d6", 1.0: "#104281"}   # ordinal: pane contact 0 / 50 / 100%
PRESSURE = {-1.5: ("Rush", AQUA), 0.0: ("No pressure", BLUE), 1.5: ("Freeze", ORANGE)}
N_PLAYERS = 16

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.markersize": 5,
})


def load(name: str, results: Path) -> list[dict]:
    with open(results / f"{name}.csv", encoding="utf-8") as f:
        return [{k: (v if k == "delay" else float(v)) for k, v in row.items()} for row in csv.DictReader(f)]


def pick(rows: list[dict], **where) -> list[dict]:
    return [r for r in rows if all(r[k] == v for k, v in where.items())]


def one(rows: list[dict], **where) -> dict:
    (row,) = pick(rows, **where)
    return row


def titled(fig, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.01, 0.915, subtitle, ha="left", fontsize=10.5, color=INK_2)


def source(fig, text: str) -> None:
    fig.text(0.01, 0.01, text, ha="left", fontsize=8.5, color=INK_2)


def fig_baseline(results: Path):
    r = load("E0_baseline", results)[0]
    pos = np.arange(1, N_PLAYERS + 1)
    sim = np.array([r[f"survival_pos_{k}"] for k in pos]) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(pos, sim, width=0.7, color=BLUE, label=f"Simulated ({int(r['games']):,} games)")
    ax.scatter(pos, exact.survival_by_position() * 100, marker="_", s=260, linewidths=2.2, color=INK,
               label="Exact odds", zorder=3)
    ax.set_xticks(pos)
    ax.set_xlabel("Position in the queue (1 = first to cross)")
    ax.set_ylabel("Chance of surviving (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left")
    titled(fig, "Ideal play: the simulator matches the math",
           f"No clock, no pushing. Mean survivors {r['mean_survivors']:.2f} ± {r['se']:.2f} vs exactly 7.00.")
    source(fig, "18 steps, 16 players. Exact odds: binomial / Markov chain (Tijms). Results: E0_baseline.csv")
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    fig.subplots_adjust(top=0.86)
    return fig


def fig_pushers(results: Path):
    rows = load("E1_pushers", results)
    n = np.arange(0, 9)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True)
    for ax, (beta, (label, _)) in zip(axes, PRESSURE.items()):
        ax.axvspan(31.25, 40, color=NEUTRAL, zorder=0)
        ax.axvspan(1.2, 4.5, color=NEUTRAL, zorder=0)
        for c, color in BLUE_RAMP.items():
            y = [one(rows, beta=beta, p_contact=c, n_pushers=k)["mean_survivors"] for k in n]
            ax.plot(n / N_PLAYERS * 100, y, "-o", color=color, label=f"{c:.0%} of pushes reveal a pane")
        ax.set_title(label, loc="left")
        ax.set_xlabel("Pushers among the 16 players (%)")
        ax.set_ylim(0, 7.6)
        ax.set_xlim(-1, 51)
    axes[0].set_ylabel("Mean survivors (of 16)")
    for ax in axes:
        ax.text(2.85, 0.25, "psychopathy\n1–5%", ha="center", fontsize=8, color=INK_2)
        ax.text(35.6, 0.25, "show + economic\ngames 31–40%", ha="center", fontsize=8, color=INK_2)
    axes[2].legend(loc="upper right", fontsize=9)
    titled(fig, "More pushers, fewer survivors, unless people freeze",
           "Pushing helps the group only when leaders freeze and most pushes reveal the next pane.")
    source(fig, "100,000 games per point; standard errors < 0.01. 16-minute clock, pushing from minute 7. "
                "Results: E1_pushers.csv")
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    return fig


def fig_breakeven(results: Path):
    rows = load("E1_pushers", results)
    n, contact = np.arange(1, 9), np.round(np.arange(0, 1.01, 0.1), 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    cmap = LinearSegmentedColormap.from_list("div", [RED, NEUTRAL, BLUE])
    norm = TwoSlopeNorm(vcenter=0, vmin=-2.0, vmax=2.0)
    for ax, (beta, (label, _)) in zip(axes, PRESSURE.items()):
        base = one(rows, beta=beta, n_pushers=0, p_contact=0.0)["mean_survivors"]
        z = np.array([[one(rows, beta=beta, n_pushers=k, p_contact=c)["mean_survivors"] - base for k in n]
                      for c in contact])
        im = ax.pcolormesh(n / N_PLAYERS * 100, contact * 100, z, cmap=cmap, norm=norm, shading="nearest",
                           edgecolors=SURFACE, linewidth=1.5)
        if z.min() < 0 < z.max():
            ax.contour(n / N_PLAYERS * 100, contact * 100, z, levels=[0], colors=INK, linewidths=1.5)
        ax.set_title(label, loc="left")
        ax.set_xlabel("Pushers among the 16 players (%)")
        ax.grid(False)
    axes[0].set_ylabel("Pushes that reveal the next pane (%)")
    cb = fig.colorbar(im, cax=fig.add_axes((0.885, 0.17, 0.015, 0.63)))
    cb.set_label("Survivors gained (+) or lost (−) vs no pushers", color=INK_2)
    cb.outline.set_visible(False)
    titled(fig, "When does pushing pay off?",
           "Only under freeze, and only when roughly 60–70% or more of pushes reveal a pane (blue, above the black line).")
    source(fig, "Change in mean survivors vs the same scenario with no pushers; 100,000 games per cell, "
                "same bridges in every cell. Results: E1_pushers.csv")
    fig.subplots_adjust(left=0.06, right=0.865, bottom=0.17, top=0.8, wspace=0.08)
    return fig


def fig_pusher_gain(results: Path):
    rows = load("E1_pushers", results)
    n = np.arange(1, 9)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    for ax, (beta, (label, _)) in zip(axes, PRESSURE.items()):
        sel = [one(rows, beta=beta, n_pushers=k, p_contact=0.5) for k in n]
        x = n / N_PLAYERS * 100
        ax.plot(x, [r["nonpusher_survival"] * 100 for r in sel], "-o", color=BLUE, label="Everyone else")
        ax.plot(x, [r["pusher_survival"] * 100 for r in sel], "-o", color=ORANGE, label="Pushers")
        ax.set_title(label, loc="left")
        ax.set_xlabel("Pushers among the 16 players (%)")
        ax.set_ylim(0, 50)
    axes[0].set_ylabel("Chance of surviving (%)")
    axes[2].legend(loc="upper right")
    titled(fig, "Pushing doesn't save the pusher",
           "Pushers survive less often than everyone else, under every kind of pressure.")
    source(fig, "Half of pushes reveal the next pane. 100,000 games per point. Results: E1_pushers.csv")
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    return fig


def fig_clock(results: Path):
    rows = load("E3_clock", results)
    clocks = sorted({r["clock"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, pushers in zip(axes, (0, 5)):
        ax.axvline(16, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(15.85, 3.9, "the show:\n16 min", fontsize=8.5, color=INK_2, ha="right", va="center")
        for beta, (label, color) in PRESSURE.items():
            y = [one(rows, beta=beta, clock=c, n_pushers=pushers)["mean_survivors"] for c in clocks]
            ax.plot(np.array(clocks) / 60, y, "-o", color=color, label=label)
        ax.set_title("No pushers" if pushers == 0 else "5 pushers (31%)", loc="left")
        ax.set_xlabel("Time limit (minutes)")
        ax.set_xlim(9.5, 25.5)
        ax.set_ylim(0, 7.6)
    axes[0].set_ylabel("Mean survivors (of 16)")
    axes[0].legend(loc="lower right")
    titled(fig, "The clock is the real killer when people freeze",
           "With time to spare almost nobody freezes to death; at 10 minutes, freezing wipes everyone out.")
    source(fig, "100,000 games per point. Half of pushes reveal a pane; pushing from minute 7. Results: E3_clock.csv")
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    return fig


FIGURES = {
    "fig1_baseline": fig_baseline,
    "fig2_pushers": fig_pushers,
    "fig3_breakeven": fig_breakeven,
    "fig4_pusher_gain": fig_pusher_gain,
    "fig5_clock": fig_clock,
}


def main(results: str | Path = "results", out: str | Path = "figures") -> None:
    results, out = Path(results), Path(out)
    out.mkdir(exist_ok=True)
    for name, make in FIGURES.items():
        fig = make(results)
        fig.savefig(out / f"{name}.png", dpi=200)
        plt.close(fig)
        print(out / f"{name}.png")


if __name__ == "__main__":
    main()
