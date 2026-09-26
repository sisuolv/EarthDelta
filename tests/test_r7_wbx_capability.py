from __future__ import annotations

import numpy as np
import pytest

from earthdelta.wbx import evaluate as ev


def test_caller_authorized_year_override_is_rejected():
    with pytest.raises(ev.YearNotAuthorized, match="caller-supplied authorized"):
        ev.assert_years_authorized([2022], authorized=frozenset({2020, 2022}))


def test_cross_year_valid_time_is_rejected_before_loader_or_array_access():
    with pytest.raises(ev.YearNotAuthorized):
        ev.evaluate(
            "/does/not/exist/prediction.zarr",
            "/does/not/exist/truth.zarr",
            init_times=[np.datetime64("2020-12-31T00", "ns")],
            lead_times=[np.timedelta64(72, "h")],
        )


def test_single_chunk_requires_issue_time_and_checks_valid_time_year():
    p = {"x": np.zeros((1, 1), dtype=np.float32)}
    t = {"x": np.zeros((1, 1), dtype=np.float32)}
    with pytest.raises(ev.YearNotAuthorized):
        ev.evaluate_single_chunk(p, t)
    with pytest.raises(ev.YearNotAuthorized):
        ev.evaluate_single_chunk(
            p, t,
            init_times=[np.datetime64("2020-12-31T00", "ns")],
            valid_times=[np.datetime64("2021-01-03T00", "ns")],
        )
