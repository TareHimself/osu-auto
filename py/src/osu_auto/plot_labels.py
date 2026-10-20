import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from osu_auto.labels import KIND_SLIDER, KIND_SPINNER, Export, load_export


def plot_path(export: Export, out: Path, window_ms: tuple[float, float] | None = None) -> None:
    labels, objects = export.labels, export.objects
    t0, t1 = window_ms or (float(labels.t_ms[0]), float(labels.t_ms[-1]))
    sel = (labels.t_ms >= t0) & (labels.t_ms <= t1)
    osel = (objects.start_ms >= t0) & (objects.start_ms <= t1)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.set_xlim(-64, 512 + 64)
    ax.set_ylim(384 + 48, -48)
    ax.set_aspect("equal")
    ax.add_patch(plt.Rectangle((0, 0), 512, 384, fill=False, linestyle="--", linewidth=0.8))

    for i in [i for i in range(len(objects)) if osel[i]]:
        colour = {KIND_SLIDER: "tab:orange", KIND_SPINNER: "tab:purple"}.get(int(objects.kind[i]), "tab:blue")
        ax.add_patch(Circle((objects.x[i], objects.y[i]), objects.radius[i], fill=False, color=colour, linewidth=1))
        if objects.kind[i] == KIND_SLIDER:
            ax.plot([objects.x[i], objects.end_x[i]], [objects.y[i], objects.end_y[i]], color=colour, linewidth=0.6, alpha=0.6)

    ax.plot(labels.x[sel], labels.y[sel], color="black", linewidth=0.7, alpha=0.8)
    pressed = sel & ((labels.press_k1 == 1) | (labels.press_k2 == 1))
    ax.scatter(labels.x[pressed], labels.y[pressed], s=12, color="red", zorder=5, label="press")

    ax.set_title(f"{export.meta['title']} [{export.meta['difficulty']}] {export.meta['source']}  {t0:.0f}..{t1:.0f} ms")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_keys(export: Export, out: Path, window_ms: tuple[float, float]) -> None:
    labels, objects = export.labels, export.objects
    t0, t1 = window_ms
    sel = (labels.t_ms >= t0) & (labels.t_ms <= t1)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.step(labels.t_ms[sel], labels.k1[sel] * 1.0, where="post", label="k1")
    ax.step(labels.t_ms[sel], labels.k2[sel] * 1.0 + 1.5, where="post", label="k2")

    for i in range(len(objects)):
        if t0 <= objects.start_ms[i] <= t1:
            ax.axvline(objects.start_ms[i], color="gray", linewidth=0.5, alpha=0.6)
            if objects.kind[i] != 0:
                ax.axvspan(objects.start_ms[i], objects.end_ms[i], color="orange", alpha=0.15)

    ax.set_xlim(t0, t1)
    ax.set_yticks([0, 1, 1.5, 2.5], ["k1 up", "k1 down", "k2 up", "k2 down"])
    ax.set_xlabel("ms")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Sanity plots for an exported label set")
    p.add_argument("export_dir")
    p.add_argument("--out", default=None, help="output directory, defaults to the export dir")
    p.add_argument("--window", type=float, nargs=2, metavar=("T0", "T1"), default=None, help="ms window for the detail plots")
    args = p.parse_args()

    export = load_export(args.export_dir)
    out = Path(args.out or args.export_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_path(export, out / "path_full.png")

    t0 = float(export.objects.start_ms[0])
    window = tuple(args.window) if args.window else (t0, t0 + 15_000)
    plot_path(export, out / "path_window.png", window)
    plot_keys(export, out / "keys_window.png", window)

    labels = export.labels
    print(f"samples={len(labels)} step={labels.step_ms:.1f}ms objects={len(export.objects)}")
    print(f"presses k1={int(labels.press_k1.sum())} k2={int(labels.press_k2.sum())} held k1={labels.k1.mean():.1%} k2={labels.k2.mean():.1%}")
    print(f"wrote {out / 'path_full.png'}, {out / 'path_window.png'}, {out / 'keys_window.png'}")


if __name__ == "__main__":
    main()
