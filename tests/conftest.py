from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _cif(pdb_id: str):
    p = FIXTURES_DIR / pdb_id / f"{pdb_id}_final.cif"
    if not p.exists():
        pytest.skip(f"CIF fixture missing — copy to {p}")
    return p


def _edia(pdb_id: str):
    p = FIXTURES_DIR / pdb_id / f"{pdb_id}_final.json"
    if not p.exists():
        pytest.skip(f"EDIA fixture missing — copy to {p}")
    return p


@pytest.fixture(scope="session")
def cif_path_5f14():
    return _cif("5f14")


@pytest.fixture(scope="session")
def cif_path_5f16():
    return _cif("5f16")


@pytest.fixture(scope="session")
def edia_path_5f16():
    return _edia("5f16")


@pytest.fixture(scope="session")
def cif_path_6ybf():
    return _cif("6ybf")


@pytest.fixture(scope="session")
def edia_path_6ybf():
    return _edia("6ybf")
