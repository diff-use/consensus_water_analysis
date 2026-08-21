import importlib.util
import sys
from pathlib import Path

import biotite.structure.io.pdbx as pdbx
import gemmi
import numpy as np
import pytest

from cw.align import align_to_reference, kabsch
from cw.io import load_protein

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "phenix" / "align_starting_model.py"


def _load_align_script():
    spec = importlib.util.spec_from_file_location("align_starting_model", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mtz(pdb_id: str):
    p = Path(__file__).parent / "fixtures" / pdb_id / f"{pdb_id}_final.mtz"
    if not p.exists():
        pytest.skip(f"MTZ fixture missing — copy to {p}")
    return p


# ── kabsch (pure math, no fixtures) ───────────────────────────────────────────


def test_kabsch_identity():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((10, 3))
    R, t = kabsch(pts, pts)
    np.testing.assert_allclose((R @ pts.T).T + t, pts, atol=1e-10)


def test_kabsch_pure_translation():
    rng = np.random.default_rng(1)
    fixed = rng.standard_normal((10, 3))
    mobile = fixed + np.array([3.0, -1.5, 2.0])
    R, t = kabsch(mobile, fixed)
    np.testing.assert_allclose((R @ mobile.T).T + t, fixed, atol=1e-10)


def test_kabsch_rotation_and_translation():
    rng = np.random.default_rng(2)
    fixed = rng.standard_normal((20, 3))
    theta = np.pi / 3
    R_true = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)],
        ]
    )
    mobile = (R_true @ fixed.T).T + np.array([1.0, -2.0, 0.5])
    R, t = kabsch(mobile, fixed)
    np.testing.assert_allclose((R @ mobile.T).T + t, fixed, atol=1e-10)


def test_kabsch_no_reflection():
    rng = np.random.default_rng(3)
    for _ in range(20):
        R, _ = kabsch(rng.standard_normal((12, 3)), rng.standard_normal((12, 3)))
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)


# ── align_to_reference integration (5f14 → 5f16) ─────────────────────────────


def _atom_counts(cif_path):
    atoms = pdbx.get_structure(pdbx.CIFFile.read(str(cif_path)), model=1, altloc="all")
    n_protein = int((~atoms.hetero).sum())
    n_water = int(((atoms.res_name == "HOH") & atoms.hetero & (atoms.element == "O")).sum())
    return n_protein, n_water


def test_align_5f16_to_6ybf(cif_path_6ybf, cif_path_5f16, tmp_path):
    ref_protein, _ = load_protein(cif_path_6ybf)
    out_path = tmp_path / "5f16_aligned.cif"

    report = align_to_reference(cif_path_5f16, ref_protein, out_path=out_path)

    assert report is not None, "alignment was skipped — too few common Cα"
    assert report["rmsd_after"] < report["rmsd_before"], (
        f"RMSD did not improve: {report['rmsd_before']} → {report['rmsd_after']}"
    )

    prot_before, water_before = _atom_counts(cif_path_5f16)
    prot_after, water_after = _atom_counts(out_path)

    assert prot_after == prot_before, f"protein atom count changed: {prot_before} → {prot_after}"
    assert water_after == water_before, f"water count changed: {water_before} → {water_after}"


# ── align_starting_model.py CLI ──────────────────────────────────────────────


def test_align_starting_model_cli(cif_path_6ybf, cif_path_5f16, tmp_path, monkeypatch, capsys):
    mod = _load_align_script()
    out = tmp_path / "5f16_aligned.cif"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "align_starting_model.py",
            "--mobile",
            str(cif_path_5f16),
            "--reference",
            str(cif_path_6ybf),
            "--out",
            str(out),
            "--pdb-id",
            "6ybf",
            "--ref-pdb-id",
            "5f16",
        ],
    )

    rc = mod.main()

    assert rc == 0
    assert out.exists(), "aligned cif was not written"
    fields = capsys.readouterr().out.strip().splitlines()[-1].split(",")
    assert fields[0] == "6ybf" and fields[1] == "5f16"
    assert fields[5] == "aligned"
    assert float(fields[4]) < float(fields[3]), "rmsd_after not less than rmsd_before"


def test_adopt_crystal_symmetry(cif_path_6ybf, cif_path_5f16, tmp_path):
    mod = _load_align_script()
    ref_protein, _ = load_protein(cif_path_6ybf)
    out = tmp_path / "5f16_aligned.cif"
    report = align_to_reference(cif_path_5f16, ref_protein, out_path=out)
    assert report is not None

    mtz_path = _mtz("6ybf")
    mod.adopt_crystal_symmetry(out, mtz_path)

    st = gemmi.read_structure(str(out))
    mtz = gemmi.read_mtz_file(str(mtz_path))
    assert st.spacegroup_hm == mtz.spacegroup.hm
    for got, want in zip(st.cell.parameters, mtz.cell.parameters, strict=True):
        assert got == pytest.approx(want, abs=1e-3)
