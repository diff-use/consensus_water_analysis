import numpy as np
import pandas as pd
import pytest
from scipy import stats

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


def test_effect_size_matches_scipys_mann_whitney_u():
    # Cliff's delta here is a hand-rolled searchsorted reimplementation of the
    # Mann-Whitney U rescaled to [-1, 1], kept because it is ~20x cheaper than
    # scipy per call on the sample sizes the bootstrap resamples. Small integer
    # ranges make ties common, which is where the left/right searchsorted
    # asymmetry has to be right.
    rng = np.random.default_rng(0)
    for _ in range(100):
        a = rng.integers(0, 5, rng.integers(2, 40)).astype(float)
        b = rng.integers(0, 5, rng.integers(2, 40)).astype(float)
        u = stats.mannwhitneyu(a, b).statistic
        assert effect_size(a, b, "mannwhitney") == pytest.approx(2 * u / (a.size * b.size) - 1)
    # sign convention: positive means the first sample sits higher
    assert effect_size(np.array([4.0, 5.0]), np.array([1.0, 2.0]), "mannwhitney") == 1.0


def test_compare_halves_unit_resampling_ignores_duplicated_rows():
    # Pseudo-replication: duplicating every water within a pdb_id adds no
    # independent information. Row resampling is fooled — the interval shrinks and
    # p collapses — while resampling whole pdb_ids is invariant to the copies.
    # One structure's left value is NaN throughout, so the unit labels have to be
    # masked in step with the values and that unit contributes to one half only.
    rng = np.random.default_rng(0)
    per_unit = rng.normal(0.0, 1.0, 30)
    left_per_unit = per_unit + 1.0
    left_per_unit[7] = np.nan
    n_boot = 400

    def compare(reps):
        units = np.repeat(np.arange(30), reps)
        left, right = np.repeat(left_per_unit, reps), np.repeat(per_unit, reps)
        return (
            compare_halves(left, right, n_boot=n_boot),
            compare_halves(left, right, n_boot=n_boot, left_units=units, right_units=units),
        )

    def width(result):
        return result["ci_high"] - result["ci_low"]

    rows_once, clustered_once = compare(1)
    rows_duplicated, clustered_duplicated = compare(16)

    assert clustered_once["n_left"] == 29  # the NaN structure dropped
    assert clustered_once["n_right"] == 30

    assert width(clustered_duplicated) == pytest.approx(width(clustered_once))
    assert clustered_duplicated["p"] == clustered_once["p"]
    # p from an inverted bootstrap is a bound, not a measurement: it floors here.
    assert clustered_once["p"] == pytest.approx(1 / n_boot)

    assert width(rows_duplicated) < width(rows_once) / 2
    assert rows_duplicated["p"] < rows_once["p"]
