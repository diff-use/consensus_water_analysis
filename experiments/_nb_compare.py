"""Diff the polished notebook against its pre-polish backup: same data, same
settings, same seed -> figures should be pixel-identical (bar deliberate label
changes) and the statistics tables cell-identical."""
import ast
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from marimo._runtime.control_flow import MarimoStopError

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

OLD, NEW = Path(sys.argv[1]), Path(sys.argv[2])
OUT = Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)


def run(path, patches=()):
    source = path.read_text()
    for before, after in patches:
        assert before in source, f"patch target missing: {before[:60]}"
        source = source.replace(before, after)
    cells = [
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and any(getattr(d, "attr", "") == "cell" for d in node.decorator_list)
    ]
    plt.close("all")
    namespace = {"__name__": "__nb__"}
    for index, cell in enumerate(cells):
        body = [n for n in cell.body if not isinstance(n, ast.Return)]
        if not body:
            continue
        code = ast.unparse(ast.Module(body=body, type_ignores=[]))
        try:
            exec(compile(code, f"<{path.name} cell {index}>", "exec"), namespace)
        except MarimoStopError:
            print(f"  [{path.name}] cell {index} stopped (mo.stop gate)")
    images = [
        np.asarray(plt.figure(n).canvas.buffer_rgba()).copy()
        for n in sorted(plt.get_fignums())
        if plt.figure(n).canvas.draw() or True
    ]
    return namespace, images


print("=== running pre-polish backup")
old_ns, old_images = run(OLD)
print("=== running polished notebook (statistics switched on to match)")
new_ns, new_images = run(NEW, patches=[
    ('value=False, label="compute per-water statistics"',
     'value=True, label="compute per-water statistics"'),
    ('value=False, label="compute per-structure statistics"',
     'value=True, label="compute per-structure statistics"'),
])

print(f"\n=== figures: {len(old_images)} old vs {len(new_images)} new")
for i, (a, b) in enumerate(zip(old_images, new_images), start=1):
    if a.shape != b.shape:
        print(f"fig{i}: SHAPE differs {a.shape} vs {b.shape}")
        continue
    diff = np.abs(a.astype(int) - b.astype(int)).max(axis=2)
    rows = np.flatnonzero(diff.any(axis=1))
    cols = np.flatnonzero(diff.any(axis=0))
    share = (diff > 0).mean()
    box = f" rows {rows.min()}-{rows.max()} cols {cols.min()}-{cols.max()}" if rows.size else ""
    print(f"fig{i}: {'IDENTICAL' if share == 0 else f'differs in {share:.4%} of pixels{box}'}")
    if share:
        plt.imsave(OUT / f"fig{i}_old.png", a)
        plt.imsave(OUT / f"fig{i}_new.png", b)


def show(name, frame):
    print(f"\n--- {name}\n{frame.to_string(index=False)}")


def compare(name, old, new):
    shared = [c for c in old.columns if c in new.columns]
    missing = [c for c in old.columns if c not in new.columns]
    added = [c for c in new.columns if c not in old.columns]
    print(f"\n=== {name}: {len(old)} vs {len(new)} rows; "
          f"dropped={missing or 'none'} added={added or 'none'}")
    same = old[shared].reset_index(drop=True).equals(new[shared].reset_index(drop=True))
    print(f"shared columns {shared}\n  -> {'IDENTICAL' if same else 'DIFFERS'}")
    if not same:
        for column in shared:
            a = old[column].reset_index(drop=True)
            b = new[column].reset_index(drop=True)
            if not a.equals(b):
                print(f"  {column}:\n    old {list(a)}\n    new {list(b)}")


water_old = old_ns["water_comparison_table"]
water_new = new_ns["water_comparison_table"]
compare("per-water table", water_old, water_new)
show("per-water (new)", water_new)

structure_old = old_ns["comparison_table"].rename(columns={"median_f1": "cutoff_f1"})
structure_old["CI"] = list(zip(structure_old.pop("CI_low"), structure_old.pop("CI_high")))
structure_new = new_ns["structure_comparison_table"]
compare("per-structure table", structure_old, structure_new)
show("per-structure (new)", structure_new)
