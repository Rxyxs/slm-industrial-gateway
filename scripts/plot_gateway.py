"""Figuras del gateway de planta a partir de los JSON de outputs/reports/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPORTS = Path("outputs/reports")
FIGS = Path("docs/figures")
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1a1a19", "#6b6a64", "#e4e3dc"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def bench() -> None:
    data = json.loads((REPORTS / "gateway_bench.json").read_text())["results"]
    sets = ["random", "high_silica"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    for k, (ax, s) in enumerate(zip(axes[:2], sets)):
        rows = [r for r in data if r["set"] == s]
        y = list(range(len(rows)))
        fail = [r["contract_failure_pct"] + r["physics_rejected_pct"] for r in rows]
        contra = [r["premise_contradicted_pct"] for r in rows]
        clean = [100 - f - c for f, c in zip(fail, contra)]
        left = [0] * len(rows)
        for vals, color, label in [(clean, AQUA, "valid, premise consistent with data"),
                                   (contra, YELLOW, "valid, premise contradicted by data"),
                                   (fail, ORANGE, "contract failure or timeout -> failsafe HOLD")]:
            ax.barh(y, vals, left=left, color=color, label=label, edgecolor="white", linewidth=2,
                    height=0.62)
            left = [l + v for l, v in zip(left, vals)]
        ax.set_yticks(y, [r["model"] for r in rows] if k == 0 else [""] * len(rows))
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.set_xlabel("% of hourly snapshots (n=60)")
        ax.set_title("Random hours" if s == "random" else "Top-decile silica hours (>= 4.14%)",
                     loc="left", color=INK, fontweight="bold")
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3, frameon=False)

    ax = axes[2]
    rows = [r for r in data if r["set"] == "random"]
    y = list(range(len(rows)))
    ax.barh([i - 0.18 for i in y], [r["latency_ms"]["p50"] for r in rows], height=0.34,
            color=BLUE, label="p50", edgecolor="white", linewidth=1)
    ax.barh([i + 0.18 for i in y], [r["latency_ms"]["p95"] for r in rows], height=0.34,
            color=ORANGE, label="p95", edgecolor="white", linewidth=1)
    ax.axvline(3000, color=MUTED, linestyle="--", linewidth=1)
    ax.set_yticks(y, [""] * len(rows))
    ax.set_xlim(0, 3600)
    ax.invert_yaxis()
    ax.set_xlabel("ms, random hours (dashed: 3 s deadline)")
    ax.set_title("Latency, RTX 4060 Ti", loc="left", color=INK, fontweight="bold")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 1.0), ncol=1, fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(FIGS / "gateway_bench.png", dpi=150, bbox_inches="tight")


def failover() -> None:
    path = REPORTS / "failover_timeline.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    tl, ev = d["timeline"], d["summary"]["events"]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    stop = next(e["t"] for e in ev if e["event"] == "engine stopped")
    start = next(e["t"] for e in ev if e["event"] == "engine started")
    ax.axvspan(stop, start, color=GRID, alpha=0.8, linewidth=0)
    ax.text((stop + start) / 2, 3150, "inference engine down", ha="center", fontsize=9, color=MUTED)
    colors = {"edge": AQUA, "cache": BLUE, "failsafe": ORANGE}
    for tier, c in colors.items():
        pts = [(r["t"], r["wall_ms"]) for r in tl if r["tier"] == tier]
        if pts:
            ax.scatter(*zip(*pts), s=36, color=c, label=tier, edgecolors="white", linewidths=1.5, zorder=3)
    ax.axhline(3000, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("seconds since start (1 request/s, real flotation hours)")
    ax.set_ylabel("response time (ms)")
    ax.set_ylim(0, 3400)
    ax.set_title("Engine outage: every request answered, none over the 3 s deadline",
                 loc="left", color=INK, fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "gateway_failover.png", dpi=150)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    bench()
    failover()
    print("figures written to", FIGS)
