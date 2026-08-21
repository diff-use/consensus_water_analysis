import pandas as pd

from cw.metrics import consensus_water_mask

# ── consensus_water_mask ──────────────────────────────────────────────────────


def _clusters(occupancies):
    return pd.DataFrame({"cluster_id": range(len(occupancies)), "cluster_occupancy": occupancies})


def test_consensus_water_mask_rejects_every_non_consensus_case():
    # cluster 0 clears the cutoff, cluster 1 does not.
    clusters = _clusters([0.8, 0.1])
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


def test_consensus_water_mask_cutoff_is_inclusive():
    clusters = _clusters([0.3])
    members = pd.DataFrame({"cluster_id": [0], "within_cutoff": [True]})
    assert bool(consensus_water_mask(members, clusters, 0.3).iloc[0])


def test_consensus_water_mask_none_clear_the_cutoff():
    clusters = _clusters([0.1, 0.2])
    members = pd.DataFrame({"cluster_id": [0, 1], "within_cutoff": [True, True]})
    assert not consensus_water_mask(members, clusters, 0.3).any()


def test_consensus_water_mask_keeps_the_member_index():
    # Callers assign the mask back onto cluster_members, so the index must align
    # even when the frame is a filtered slice.
    clusters = _clusters([0.8])
    members = pd.DataFrame({"cluster_id": [0, 0], "within_cutoff": [True, False]}, index=[5, 9])
    assert list(consensus_water_mask(members, clusters, 0.3).index) == [5, 9]
