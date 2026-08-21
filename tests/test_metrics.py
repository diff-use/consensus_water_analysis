import pandas as pd

from cw.metrics import consensus_water_mask

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
