"""Self-exclusion in neighbour lookup must be by identity, never by distance.

Regression test for a latent bug in the served path: `view_neighbors` used to
drop any bank cell whose weight was >= 0.999999, treating "distance 0" as a
proxy for "this is the query cell itself". The proxy is wrong whenever two
distinct cells share coordinates in a view — which is routine, not exotic:

  * the CONDITION view is (c_rate_1, c_rate_2, soc_transition_pct), so every
    pair of cells on the same charge policy sits at distance 0;
  * a whole study can share one charge policy (HUST's 77 cells all use
    5C(80%)-1C), so adding that study to the bank would have made every one of
    its cells invisible to every other one.

The correct rule, and the one the leakage discipline actually states, is that a
cell may not be its own neighbour — an identity check.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.common import load_artifacts, view_neighbors


@pytest.fixture()
def art():
    a, err = load_artifacts()
    if a is None:
        pytest.skip(f"artifacts not built: {err}")
    return a


def _twin(art, source_id: str, new_id: str) -> dict:
    """A copy of `art` whose bank gains an exact duplicate of one cell.

    The twin is a distinct cell at distance 0 from the original in EVERY view —
    the hardest case for a distance-based self-check.
    """
    bank = art["bank"]
    row = bank[bank["cell_id"] == source_id].iloc[0].copy()
    row["cell_id"] = new_id
    return {**art, "bank": pd.concat([bank, row.to_frame().T], ignore_index=True)}


def test_zero_distance_twin_is_kept_when_it_is_not_the_query(art):
    """A bank cell identical to the query is a real neighbour unless it IS the query."""
    meta = art["meta"]
    src = art["bank"]["cell_id"].iloc[0]
    a2 = _twin(art, src, "TWIN_OF_" + src)
    row = a2["bank"][a2["bank"]["cell_id"] == src].iloc[0]
    query = {c: float(row[c]) for c in meta["base_features"]}

    nb = view_neighbors(a2, "condition", query, exclude_group=None,
                        exclude_cell_id=src)
    assert "TWIN_OF_" + src in set(nb["cell_id"]), \
        "a distinct cell at distance 0 was dropped as if it were the query"
    assert src not in set(nb["cell_id"]), "the query cell was used as its own neighbour"
    # distance 0 -> weight exactly 1.0; the old proxy filtered precisely this
    assert nb.loc[nb["cell_id"] == "TWIN_OF_" + src, "weight"].iloc[0] == pytest.approx(1.0)


def test_shared_policy_neighbours_survive_in_the_condition_view(art):
    """Same-policy cells are coincident in the condition view and must be kept.

    This is the mechanism behind the cross-study failure: without it, a bank
    whose members share a charge policy supplies each of them with zero
    same-policy neighbours.
    """
    meta, bank = art["meta"], art["bank"]
    grp = bank.groupby("policy_group_id").size()
    shared = grp[grp >= 2]
    if shared.empty:
        pytest.skip("no policy group with >= 2 bank cells")
    gid = shared.index[0]
    members = bank[bank["policy_group_id"] == gid]["cell_id"].tolist()
    row = bank[bank["cell_id"] == members[0]].iloc[0]
    query = {c: float(row[c]) for c in meta["base_features"]}

    nb = view_neighbors(art, "condition", query, exclude_group=None,
                        exclude_cell_id=members[0])
    assert set(members[1:]) & set(nb["cell_id"]), \
        "same-policy siblings vanished from the condition view"


def test_self_is_excluded_only_by_identity(art):
    """Passing the query's own id removes it; passing nothing keeps it."""
    meta, bank = art["meta"], art["bank"]
    cid = bank["cell_id"].iloc[0]
    row = bank[bank["cell_id"] == cid].iloc[0]
    query = {c: float(row[c]) for c in meta["base_features"]}

    without = view_neighbors(art, "behavior", query, exclude_group=None,
                             exclude_cell_id=cid)
    assert cid not in set(without["cell_id"])

    with_self = view_neighbors(art, "behavior", query, exclude_group=None)
    assert cid in set(with_self["cell_id"]), \
        "identity exclusion must be explicit, not inferred from distance"
    assert with_self["weight"].iloc[0] == pytest.approx(1.0)


def test_group_exclusion_still_removes_the_cell_itself(art):
    """The grouped-CV path stays correct: own group excluded implies self excluded."""
    meta, bank = art["meta"], art["bank"]
    row = bank.iloc[0]
    query = {c: float(row[c]) for c in meta["base_features"]}
    for view in ("condition", "behavior"):
        nb = view_neighbors(art, view, query, exclude_group=row["policy_group_id"])
        assert row["cell_id"] not in set(nb["cell_id"])
        assert (nb["policy_group_id"] != row["policy_group_id"]).all()


def test_coverage_is_not_silently_deflated_by_the_old_proxy(art):
    """Coverage over a bank containing coincident cells must count them."""
    meta = art["meta"]
    src = art["bank"]["cell_id"].iloc[0]
    a2 = _twin(art, src, "TWIN_OF_" + src)
    row = a2["bank"][a2["bank"]["cell_id"] == src].iloc[0]
    query = {c: float(row[c]) for c in meta["base_features"]}

    cov_before = view_neighbors(art, "condition", query, exclude_group=None,
                                exclude_cell_id=src)["weight"].sum()
    cov_after = view_neighbors(a2, "condition", query, exclude_group=None,
                               exclude_cell_id=src)["weight"].sum()
    assert cov_after > cov_before, \
        "adding a coincident cell to the bank must raise coverage, not leave it flat"
