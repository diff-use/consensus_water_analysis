import numpy as np
import pandas as pd
import pytest

from cw.metrics import (
    compare_halves,
    consensus_water_mask,
    effect_size,
)

# ── consensus_water_mask ──────────────────────────────────────────────────────


def test_consensus_water_mask_rejects_every_non_consensus_case():
    # cluster 0 clears the cutoff, cluster 1 does not.
    clusters = pd.DataFrame({"cluster_id": [0, 1], "cluster_occupancy": [0.8, 0.1]})
    members = pd.DataFrame(
        {
            "cluster_id": [0, 0, 1, -1],
            "within_cutoff": [True, False, True, False],
        }
    )
    # consensus / radius-rejected / low-occupancy cluster / noise
    assert list(consensus_water_mask(members, clusters, 0.3)) == [
        True,
        False,
        False,
        False,
    ]


# ── compare_halves ────────────────────────────────────────────────────────────


def test_effect_size_spans_full_separation():
    low, high = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
    # Cliff's delta: +1 when every a beats every b, -1 reversed, 0 when identical.
    assert effect_size(high, low, "mannwhitney") == 1.0
    assert effect_size(low, high, "mannwhitney") == -1.0
    assert effect_size(low, low, "mannwhitney") == 0.0


def test_compare_halves_bootstrap_ci_brackets_the_effect():
    rng = np.random.default_rng(0)
    left, right = rng.normal(1.0, 1.0, 200), rng.normal(0.0, 1.0, 200)
    result = compare_halves(left, right, n_boot=200)
    assert result["n_left"] == 200 and result["n_right"] == 200
    assert result["effect"] > 0  # left sits higher
    assert result["ci_low"] < result["effect"] < result["ci_high"]
    assert result["p"] < 0.001


def test_compare_halves_unit_resampling_ignores_duplicated_rows():
    # Pseudo-replication: duplicating every water within a pdb_id adds no
    # independent information. Row resampling is fooled — the interval shrinks and
    # p collapses — while resampling whole pdb_ids is invariant to the copies.
    rng = np.random.default_rng(0)
    per_unit = rng.normal(0.0, 1.0, 30)
    n_boot = 400

    def compare(reps):
        units = np.repeat(np.arange(30), reps)
        left, right = np.repeat(per_unit + 1.0, reps), np.repeat(per_unit, reps)
        return (
            compare_halves(left, right, n_boot=n_boot),
            compare_halves(left, right, n_boot=n_boot, left_units=units, right_units=units),
        )

    def width(result):
        return result["ci_high"] - result["ci_low"]

    rows_once, clustered_once = compare(1)
    rows_duplicated, clustered_duplicated = compare(16)

    assert width(clustered_duplicated) == pytest.approx(width(clustered_once))
    assert clustered_duplicated["p"] == clustered_once["p"]
    # p from an inverted bootstrap is a bound, not a measurement: it floors here.
    assert clustered_once["p"] == pytest.approx(1 / n_boot)

    assert width(rows_duplicated) < width(rows_once) / 2
    assert rows_duplicated["p"] < rows_once["p"]
