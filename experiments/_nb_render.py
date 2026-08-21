"""Headless marimo render: exec each @app.cell body in file order in one shared
namespace, then save every open figure. marimo 0.23.9's app.run()/app.embed()
exit 0 without executing a cell, so this is the only way to smoke-test a
notebook without a browser."""
import ast
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from marimo._runtime.control_flow import MarimoStopError

NOTEBOOK = Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks/metrics_violin.py")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "experiments/_nb_render_out")
OUT.mkdir(parents=True, exist_ok=True)

tree = ast.parse(NOTEBOOK.read_text())
namespace = {"__name__": "__nb__"}
cells = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and any(getattr(d, "attr", "") == "cell" for d in node.decorator_list)
]
print(f"{len(cells)} cells")
for index, cell in enumerate(cells):
    body = [n for n in cell.body if not isinstance(n, ast.Return)]
    if not body:
        continue
    source = ast.unparse(ast.Module(body=body, type_ignores=[]))
    try:
        exec(compile(source, f"<cell {index}>", "exec"), namespace)
    except MarimoStopError:
        print(f"cell {index} stopped (mo.stop gate)")
    except Exception as exc:
        print(f"CELL {index} FAILED ({type(exc).__name__}): {exc}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)

for number in plt.get_fignums():
    path = OUT / f"fig{number}.png"
    plt.figure(number).savefig(path, dpi=110, bbox_inches="tight")
    print(f"wrote {path}")
print("all cells ok")
