"""Plot a 2D reciprocal-space plane of reflections from an MTZ file.

The user fixes one Miller index (h, k, or l) to a constant (default 0) and the
script scatters the reflections lying in that plane: each point sits at the two
free indices, its marker size scales with a structure-factor amplitude column,
and its colour encodes the matching phase column (grey when no phase exists).

Columns are auto-detected from the MTZ column *types* (F = amplitude,
P = phase), which is robust across the pdb-redo (FP/FWT/FC_ALL ...) and phenix
(F-obs/2FOFCWT/F-model ...) naming conventions. Both can be overridden.

One-off / exploratory — lives in experiments/ (gitignored).

Examples
--------
    python experiments/mtz_reflection_plane.py data/hewls_pdb_redo/3atn/3atn_final.mtz
    python experiments/mtz_reflection_plane.py data/.../3atn_final.mtz --plane h=0 --amplitude FP
    python experiments/mtz_reflection_plane.py data/.../3atn_refined_by_3atn_auto_001.mtz --plane l
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import reciprocalspaceship as rs
from matplotlib.ticker import MaxNLocator

# Amplitude auto-detection preference. Earlier patterns win. We prefer a phased
# map coefficient (2Fo-Fc) so the phase-colouring is exercised by default, then
# the calculated model amplitude, then the raw observed amplitude (which has no
# companion phase and therefore plots grey).
AMPLITUDE_PREFERENCE = [
    "FWT", "2FOFCWT",              # 2Fo-Fc map coefficient (phased)
    "FC_ALL", "F-model", "FCALC", "FC",  # calculated (phased)
    "FP", "F-obs", "FOBS", "FOSC", "F",  # observed (no phase)
]

AXES = ("H", "K", "L")


def parse_plane(text: str) -> tuple[str, int]:
    """Parse '--plane' into (axis, value). Accepts 'l', 'L', 'l=0', 'h=2'."""
    text = text.strip()
    if "=" in text:
        axis, _, value = text.partition("=")
        value = int(value)
    else:
        axis, value = text, 0
    axis = axis.strip().upper()
    if axis not in AXES:
        raise argparse.ArgumentTypeError(f"plane axis must be one of h/k/l, got {axis!r}")
    return axis, value


def _mtztype(dataset: rs.DataSet, column: str) -> str:
    return dataset[column].dtype.mtztype


def pick_amplitude(dataset: rs.DataSet) -> str:
    f_columns = [c for c in dataset.columns if _mtztype(dataset, c) == "F"]
    if not f_columns:
        raise SystemExit("No amplitude (type F) column found in the MTZ.")
    upper = {c.upper(): c for c in f_columns}
    for preferred in AMPLITUDE_PREFERENCE:
        if preferred.upper() in upper:
            return upper[preferred.upper()]
    return f_columns[0]


def pick_phase(dataset: rs.DataSet, amplitude: str) -> str | None:
    """Phase companion = first type-P column following the amplitude in MTZ
    order, provided no other amplitude column comes between them."""
    columns = list(dataset.columns)
    start = columns.index(amplitude)
    for column in columns[start + 1:]:
        mtype = _mtztype(dataset, column)
        if mtype == "F":
            break
        if mtype == "P":
            return column
    return None


def add_friedel(dataset, amplitude: str, phase: str | None):
    """Append Friedel mates (-h,-k,-l): amplitude equal, phase negated. Needed
    to fill the full plane for non-centrosymmetric space groups, whose P1
    expansion only covers a hemisphere."""
    mate = dataset.reset_index()
    mate["H"], mate["K"], mate["L"] = -mate["H"], -mate["K"], -mate["L"]
    if phase is not None:
        mate[phase] = (-mate[phase]) % 360.0
    mate = mate.set_index(["H", "K", "L"])
    combined = rs.concat([dataset, mate])
    return combined[~combined.index.duplicated(keep="first")]


def build_plane(mtz_path: Path, amplitude: str, phase: str | None,
                axis: str, value: int, expand: bool, friedel: bool) -> rs.DataSet:
    columns = [amplitude] + ([phase] if phase else [])
    dataset = rs.read_mtz(str(mtz_path))[columns].copy()
    dataset = dataset.dropna()
    if expand:
        # expand_to_p1 applies the space-group symmetry ops and transforms
        # Phase-dtype columns correctly. It does NOT add Friedel mates, so for
        # chiral space groups the result is a hemisphere.
        dataset = dataset.expand_to_p1()
    if friedel:
        dataset = add_friedel(dataset, amplitude, phase)
    dataset = dataset.reset_index()
    return dataset[dataset[axis] == value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mtz", type=Path, help="input MTZ file")
    parser.add_argument("--plane", type=parse_plane, default=("L", 0),
                        help="fixed axis and value, e.g. 'l', 'h=0', 'k=2' (default l=0)")
    parser.add_argument("--amplitude", help="amplitude column (default: auto-detect)")
    parser.add_argument("--phase", help="phase column (default: auto-detect companion)")
    parser.add_argument("--no-phase", action="store_true",
                        help="ignore phase; plot everything grey")
    parser.add_argument("--no-expand", action="store_true",
                        help="do not expand to P1 (show only the stored ASU reflections)")
    parser.add_argument("--no-friedel", action="store_true",
                        help="do not add Friedel mates (leaves a hemisphere for chiral space groups)")
    parser.add_argument("--lim", type=int,
                        help="cap both free axes to [-LIM, LIM] (e.g. --lim 20); "
                             "reflections outside are dropped and marker scaling recomputed")
    parser.add_argument("--size-scale", choices=["linear", "sqrt", "log"], default="sqrt",
                        help="amplitude -> marker-area mapping (default sqrt; log tames strong reflections most)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="marker-size multiplier (default 1.0)")
    parser.add_argument("--cmap", default="twilight",
                        help="cyclic colormap for phase (default twilight)")
    parser.add_argument("--out", type=Path,
                        help="output image path (default: <mtz stem>_<plane>.png beside the MTZ)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--transparent", action="store_true",
                        help="save with a transparent background")
    parser.add_argument("--no-colorbar", action="store_true",
                        help="omit the phase colorbar")
    args = parser.parse_args()

    axis, value = args.plane
    free_axes = [a for a in AXES if a != axis]

    probe = rs.read_mtz(str(args.mtz))
    f_columns = [c for c in probe.columns if _mtztype(probe, c) == "F"]
    p_columns = [c for c in probe.columns if _mtztype(probe, c) == "P"]

    amplitude = args.amplitude or pick_amplitude(probe)
    if amplitude not in probe.columns:
        raise SystemExit(f"amplitude column {amplitude!r} not in MTZ. Available F columns: {f_columns}")

    if args.no_phase:
        phase = None
    elif args.phase:
        phase = args.phase
        if phase not in probe.columns:
            raise SystemExit(f"phase column {phase!r} not in MTZ. Available P columns: {p_columns}")
    else:
        phase = pick_phase(probe, amplitude)

    print(f"MTZ:        {args.mtz}")
    print(f"spacegroup: {probe.spacegroup.short_name()}")
    print(f"plane:      {axis} = {value}  (free axes: {free_axes[0]}, {free_axes[1]})")
    print(f"amplitude:  {amplitude}   (F columns: {f_columns})")
    print(f"phase:      {phase or 'none -> grey'}   (P columns: {p_columns})")

    plane = build_plane(args.mtz, amplitude, phase, axis, value,
                        expand=not args.no_expand, friedel=not args.no_friedel)
    if plane.empty:
        raise SystemExit(f"No reflections in plane {axis}={value}.")

    if args.lim is not None:
        plane = plane[(plane[free_axes[0]].abs() <= args.lim)
                      & (plane[free_axes[1]].abs() <= args.lim)]
        if plane.empty:
            raise SystemExit(f"No reflections within |index| <= {args.lim}.")

    xs = plane[free_axes[0]].to_numpy()
    ys = plane[free_axes[1]].to_numpy()
    amp = plane[amplitude].to_numpy(dtype=float)

    # Normalise to the 99th percentile so a few strong reflections do not swamp
    # the plot, then compress with the chosen scale (sqrt/log tame the strong
    # tail; area still grows with amplitude). Floor keeps weak reflections visible.
    reference = np.percentile(amp, 99) or amp.max() or 1.0
    norm = np.clip(amp / reference, 0.0, None)
    if args.size_scale == "linear":
        weight = norm
    elif args.size_scale == "sqrt":
        weight = np.sqrt(norm)
    else:  # log
        weight = np.log1p(norm) / np.log1p(1.0)
    sizes = np.clip(weight, 0.0, 1.5) * (80.0 * args.scale) + 1.5

    grey = "0.4"
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    if phase is not None:
        colors = plane[phase].to_numpy(dtype=float) % 360.0
        # Grey outline so light phase colors stay visible against the background.
        scatter = ax.scatter(xs, ys, s=sizes, c=colors, cmap=args.cmap,
                             vmin=0, vmax=360, edgecolors=grey, linewidths=0.4, alpha=0.85)
        if not args.no_colorbar:
            cbar = fig.colorbar(scatter, ax=ax, label=f"phase {phase} (deg)")
            cbar.set_ticks([0, 90, 180, 270, 360])
    else:
        ax.scatter(xs, ys, s=sizes, color=grey, linewidths=0, alpha=0.85)

    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel(free_axes[0])
    ax.set_ylabel(free_axes[1])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_aspect("equal")
    ax.set_title(f"{axis}={value} plane · size ∝ {amplitude}"
                 + (f" · colour = {phase}" if phase else ""))

    out = args.out or args.mtz.with_name(f"{args.mtz.stem}_{axis}{value}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi, transparent=args.transparent)
    print(f"wrote:      {out}  ({len(plane)} reflections)")


if __name__ == "__main__":
    main()
