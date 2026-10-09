"""Article figures from the saved sweep results (results/*.csv). Run:

    python -m glass_bridge.figures            # writes figures/*.png

Style follows *Storytelling with Data* (Knaflic): the title states the takeaway, context is
gray, one accent color carries the story, lines are labeled directly instead of with legends,
and the key point is annotated on the chart.
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

# Reference palette, light mode (Medium pages are white). Gray = context; orange = hesitation
# (the story); blue = what helps.
SURFACE, INK, INK_2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#ecebe7"
GRAY, GRAY_LIGHT, BAND = "#7a7974", "#b9b8b3", "#f3f2ef"
ORANGE, BLUE, RED, NEUTRAL = "#eb6834", "#2a78d6", "#e34948", "#f0efec"
FREEZE, NONE, RUSH = 1.5, 0.0, -1.5
PRESSURE = {RUSH: "Rush", NONE: "No pressure", FREEZE: "Freeze"}
N_PLAYERS = 16
SHOW_CLOCK = 960.0

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.edgecolor": GRAY_LIGHT, "axes.labelcolor": INK_2, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "ytick.left": False, "legend.frameon": False, "lines.linewidth": 2.2,
})


# ---- Data -----------------------------------------------------------------------------------

def load(name: str, results: Path) -> list[dict]:
    with open(results / f"{name}.csv", encoding="utf-8") as f:
        return [{k: (v if k == "delay" else float(v)) for k, v in row.items()} for row in csv.DictReader(f)]


def one(rows: list[dict], **where) -> dict:
    (row,) = [r for r in rows if all(r[k] == v for k, v in where.items())]
    return row


def cost_per_pusher(rows: list[dict], **where) -> float:
    """Slope of mean survivors vs number of pushers (0-8): survivors gained per extra pusher."""
    n = np.arange(9)
    return float(np.polyfit(n, [one(rows, n_pushers=k, **where)["mean_survivors"] for k in n], 1)[0])


# ---- Layout helpers -------------------------------------------------------------------------

def headline(fig, title: str, subtitle: str, source: str) -> None:
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.012, 0.905, subtitle, ha="left", va="top", fontsize=10.5, color=INK_2, linespacing=1.4)
    fig.text(0.012, 0.012, source, ha="left", va="bottom", fontsize=8.5, color=GRAY)


def end_labels(ax, items: list[tuple[float, float, str, str]], min_gap: float, dx: float) -> None:
    """Label lines at their right end, nudging labels apart vertically. items: (x, y, text, color)."""
    items = sorted(items, key=lambda it: it[1])
    ys = [it[1] for it in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    for (x, _, text, color), y in zip(items, ys):
        ax.text(x + dx, y, text, color=color, va="center", ha="left", fontsize=10, fontweight="bold",
                clip_on=False)


def pct_axis(ax) -> None:
    ax.set_xlim(0, 50)
    ax.set_xticks([0, 10, 20, 30, 40, 50], ["0%", "10%", "20%", "30%", "40%", "50%"])


# ---- Figures --------------------------------------------------------------------------------

def fig_baseline(results: Path):
    r = load("E0_baseline", results)[0]
    pos = np.arange(1, N_PLAYERS + 1)
    sim = np.array([r[f"survival_pos_{k}"] for k in pos]) * 100
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.bar(pos, sim, width=0.7, color=BLUE)
    ax.scatter(pos, exact.survival_by_position() * 100, marker="_", s=300, linewidths=2.4, color=INK, zorder=3)
    ax.annotate("— exact odds", (16.45, 100), xytext=(0, 0), textcoords="offset points", va="center",
                fontsize=10, color=INK, fontweight="bold", annotation_clip=False)
    ax.annotate("simulated", (16.45, 90), va="center", fontsize=10, color=BLUE, fontweight="bold",
                annotation_clip=False)
    ax.set_xticks(pos)
    ax.set_xlabel("Position in the queue (1 = first to step onto the bridge)")
    ax.set_ylabel("Chance of surviving")
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlim(0.4, 16.6)
    headline(fig, "The simulator matches the math",
             f"With ideal play (no clock, no pushing), {r['mean_survivors']:.2f} of 16 players survive on average; "
             "the exact answer is 7.\nYour odds depend almost entirely on where you stand in the queue.",
             f"{int(r['games']):,} simulated games. Exact odds: binomial / Markov chain (Tijms). "
             "Results: E0_baseline.csv")
    fig.subplots_adjust(left=0.08, right=0.88, top=0.78, bottom=0.14)
    return fig


def fig_pushers(results: Path):
    rows = load("E1_pushers", results)
    n = np.arange(9)
    x = n / N_PLAYERS * 100
    lines = [  # (pressure, contact, color, style, label)
        (RUSH, 0.5, GRAY, "-", "Rush"),
        (NONE, 0.5, GRAY_LIGHT, "-", "No pressure"),
        (FREEZE, 0.5, ORANGE, "-", "Freeze"),
        (FREEZE, 1.0, ORANGE, (0, (4, 2)), "Freeze, every push\nreveals the next pane"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.axvspan(31.25, 40, color=BAND, zorder=0)
    ax.text(35.6, 0.15, "the show and\neconomic games\n(31–40%)", ha="center", va="bottom", fontsize=8.5, color=GRAY)
    labels = []
    for beta, c, color, style, label in lines:
        y = [one(rows, beta=beta, p_contact=c, n_pushers=k)["mean_survivors"] for k in n]
        ax.plot(x, y, color=color, linestyle=style)
        slope = cost_per_pusher(rows, beta=beta, p_contact=c)
        labels.append((x[-1], y[-1], f"{label}\n{slope:+.2f} per pusher", color))
    end_labels(ax, labels, min_gap=0.85, dx=1.0)
    drop = (one(rows, beta=NONE, p_contact=0.5, n_pushers=0)["mean_survivors"]
            - one(rows, beta=FREEZE, p_contact=0.5, n_pushers=0)["mean_survivors"])
    ax.annotate(f"Freezing alone costs\n{drop:.1f} survivors,\nbefore anyone pushes",
                xy=(0, one(rows, beta=FREEZE, p_contact=0.5, n_pushers=0)["mean_survivors"]),
                xytext=(4, 4.0), fontsize=9.5, color=ORANGE, va="center",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1))
    pct_axis(ax)
    ax.set_ylim(0, 7.5)
    ax.set_xlabel("Pushers among the 16 players")
    ax.set_ylabel("Survivors, on average")
    headline(fig, "Hesitating costs more lives than pushing",
             "Every pusher lowers the number of survivors. The one exception: a frozen group, when each push\n"
             "lands on the next pane and shows everyone where to step. Half of pushes reveal a pane unless noted.",
             "100,000 games per point; 16-minute clock, pushing from minute 7. "
             "\"Per pusher\" = slope of the line. Results: E1_pushers.csv")
    fig.subplots_adjust(left=0.07, right=0.76, top=0.77, bottom=0.13)
    return fig


def fig_breakeven(results: Path):
    rows = load("E1_pushers", results)
    n, contact = np.arange(1, 9), np.round(np.arange(0, 1.01, 0.1), 1)
    x, y = n / N_PLAYERS * 100, contact * 100
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    cmap = LinearSegmentedColormap.from_list("div", [RED, NEUTRAL, BLUE])
    norm = TwoSlopeNorm(vcenter=0, vmin=-2.0, vmax=2.0)
    for ax, beta in zip(axes, (RUSH, NONE, FREEZE)):
        base = one(rows, beta=beta, n_pushers=0, p_contact=0.0)["mean_survivors"]
        z = np.array([[one(rows, beta=beta, n_pushers=k, p_contact=c)["mean_survivors"] - base for k in n]
                      for c in contact])
        im = ax.pcolormesh(x, y, z, cmap=cmap, norm=norm, shading="nearest", edgecolors=SURFACE, linewidth=1.5)
        outline_positive(ax, x, y, z)
        ax.set_title(PRESSURE[beta], loc="left", color=ORANGE if beta == FREEZE else INK)
        ax.set_xticks([10, 20, 30, 40, 50], ["10%", "20%", "30%", "40%", "50%"])
        ax.set_xlabel("Pushers among the 16 players")
        ax.grid(False)
        ax.tick_params(left=False, bottom=False)
        if (z > 0).any():
            ax.text(x.mean(), 108, "pushing helps here", ha="center", va="bottom", fontsize=9.5,
                    color=BLUE, fontweight="bold", clip_on=False)
    axes[0].set_yticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
    axes[0].set_ylabel("Pushes that reveal the next pane")
    cb = fig.colorbar(im, cax=fig.add_axes((0.905, 0.16, 0.012, 0.6)))
    cb.set_label("Survivors gained (+) or lost (−)\nvs the same group with no pushers", color=INK_2)
    cb.outline.set_visible(False)
    headline(fig, "When does pushing pay off?",
             "Only when the group is frozen and most pushes (about 60–70% or more) reveal the next pane.\n"
             "Everywhere else, every extra pusher costs lives.",
             "Change in mean survivors vs no pushers; 100,000 games per cell, same bridges in every cell. "
             "Results: E1_pushers.csv")
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.16, top=0.72, wspace=0.08)
    return fig


def outline_positive(ax, x, y, z) -> None:
    """Draw the outer border of the cells where z > 0."""
    pos = z > 0
    hx, hy = (x[1] - x[0]) / 2, (y[1] - y[0]) / 2
    rows, cols = pos.shape
    inside = lambda i, j: 0 <= i < rows and 0 <= j < cols and pos[i, j]
    for i in range(rows):
        for j in range(cols):
            if not pos[i, j]:
                continue
            cx, cy = x[j], y[i]
            for di, dj, seg in ((1, 0, ((cx - hx, cx + hx), (cy + hy, cy + hy))),
                                (-1, 0, ((cx - hx, cx + hx), (cy - hy, cy - hy))),
                                (0, 1, ((cx + hx, cx + hx), (cy - hy, cy + hy))),
                                (0, -1, ((cx - hx, cx - hx), (cy - hy, cy + hy)))):
                if not inside(i + di, j + dj):
                    ax.plot(*seg, color=BLUE, linewidth=2.5, solid_capstyle="projecting", clip_on=False)


def fig_pusher_gain(results: Path):
    rows = load("E1_pushers", results)
    n = np.arange(1, 9)
    x = n / N_PLAYERS * 100
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    for ax, beta in zip(axes, (RUSH, NONE, FREEZE)):
        sel = [one(rows, beta=beta, n_pushers=k, p_contact=0.5) for k in n]
        others = [r["nonpusher_survival"] * 100 for r in sel]
        pushers = [r["pusher_survival"] * 100 for r in sel]
        ax.plot(x, others, color=GRAY_LIGHT)
        ax.plot(x, pushers, color=ORANGE)
        ax.set_title(PRESSURE[beta], loc="left")
        ax.set_xlabel("Pushers among the 16 players")
        ax.set_xticks([10, 30, 50], ["10%", "30%", "50%"])
        ax.set_xlim(6, 50)
        ax.set_ylim(0, 50)
        if beta == RUSH:
            ax.text(x[0], others[0] + 2.5, "everyone else", color=GRAY, fontsize=10, fontweight="bold")
            ax.text(x[0], pushers[0] - 4.5, "pushers", color=ORANGE, fontsize=10, fontweight="bold")
    axes[0].set_yticks([0, 10, 20, 30, 40, 50], ["0%", "10%", "20%", "30%", "40%", "50%"])
    axes[0].set_ylabel("Chance of surviving")
    headline(fig, "Pushing doesn't save the pusher",
             "Pushers survive less often than everyone else, under every kind of pressure.",
             "Half of pushes reveal the next pane. 100,000 games per point. Results: E1_pushers.csv")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.78, wspace=0.08)
    return fig


def fig_clock(results: Path):
    rows = load("E3_clock", results)
    clocks = np.array(sorted({r["clock"] for r in rows}))
    style = {RUSH: BLUE, NONE: GRAY_LIGHT, FREEZE: ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    for ax, pushers in zip(axes, (0, 5)):
        ax.axvline(16, color=GRAY_LIGHT, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(16.15, 0.2, "the show:\n16 minutes", fontsize=8.5, color=GRAY, va="bottom")
        labels = []
        for beta, color in style.items():
            y = [one(rows, beta=beta, clock=c, n_pushers=pushers)["mean_survivors"] for c in clocks]
            ax.plot(clocks / 60, y, color=color)
            labels.append((clocks[-1] / 60, y[-1], PRESSURE[beta], color if color != GRAY_LIGHT else GRAY))
        end_labels(ax, labels, min_gap=0.45, dx=0.3)
        r = one(rows, beta=FREEZE, clock=SHOW_CLOCK, n_pushers=pushers)
        ax.plot([16], [r["mean_survivors"]], "o", color=ORANGE, markersize=7)
        ax.annotate(f"clock runs out in\n{r['p_clock_ran_out']:.0%} of games",
                    (16, r["mean_survivors"]), xytext=(17.2, 1.2), fontsize=9.5, color=ORANGE,
                    arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1))
        ax.set_title("No pushers" if pushers == 0 else "5 pushers (31%, like the show)", loc="left")
        ax.set_xlabel("Time limit (minutes)")
        ax.set_xlim(10, 25)
        ax.set_ylim(0, 7.5)
    axes[0].set_ylabel("Survivors, on average")
    rush = one(rows, beta=RUSH, clock=SHOW_CLOCK, n_pushers=0)
    headline(fig, "The clock only kills those who hesitate",
             f"Players who decide quickly beat the clock in {1 - rush['p_clock_ran_out']:.1%} of games and keep "
             "the math's 7 survivors.\nFrozen players run out of time in most games at the show's 16 minutes.",
             "100,000 games per point. With pushers, half of pushes reveal a pane; pushing from minute 7. "
             "Results: E3_clock.csv")
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.14, top=0.76, wspace=0.34)
    return fig


def fig_distribution(results: Path):
    rows = load("E3_clock", results)
    k = np.arange(N_PLAYERS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    for ax, beta in zip(axes, (RUSH, NONE, FREEZE)):
        r = one(rows, beta=beta, clock=SHOW_CLOCK, n_pushers=0)
        p = np.array([r[f"p_{i}_survivors"] for i in k]) * 100
        colors = [ORANGE if (beta == FREEZE and i == 0) else GRAY_LIGHT for i in k]
        ax.bar(k, p, width=0.75, color=colors)
        ax.set_title(f"{PRESSURE[beta]}: {r['mean_survivors']:.1f} survivors on average", loc="left",
                     color=ORANGE if beta == FREEZE else INK)
        ax.set_xticks([0, 3, 7, 10, 16])
        ax.set_xlabel("Survivors in one game")
        if beta == FREEZE:
            ax.annotate(f"nobody survives\nin {p[0]:.0f}% of games", (0.5, p[0] * 0.95), xytext=(3, p[0] * 0.8),
                        fontsize=9.5, color=ORANGE, fontweight="bold", va="top")
    axes[0].set_yticks([0, 20, 40, 60], ["0%", "20%", "40%", "60%"])
    axes[0].set_ylabel("Share of games")
    headline(fig, "A frozen group usually loses everyone",
             "The average hides the spread: freezing doesn't trim the survivors, it wipes out whole games.",
             "16-minute clock, no pushers. 100,000 games each. Results: E3_clock.csv")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.78, wspace=0.08)
    return fig


FIGURES = {
    "fig1_baseline": fig_baseline,
    "fig2_pushers": fig_pushers,
    "fig3_breakeven": fig_breakeven,
    "fig4_pusher_gain": fig_pusher_gain,
    "fig5_clock": fig_clock,
    "fig6_distribution": fig_distribution,
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
