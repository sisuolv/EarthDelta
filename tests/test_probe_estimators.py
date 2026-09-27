import numpy as np

from earthdelta.probe import per_issue_oracle, static_best_arm, ttt_oracle


def test_oracle_static_and_ttt_hand_case():
    # arm 0 is F0; arm 1 is good on both issues; arm 2 is a bad random control.
    values = np.array([
        [[1.0, 1.0], [0.5, 0.5], [1.2, 1.2]],
        [[1.0, 1.0], [0.6, 0.6], [1.1, 1.1]],
    ])
    oracle = per_issue_oracle(values)
    assert oracle["selected_arm"] == [1, 1]
    assert oracle["selected_f0_fraction"] == 0.0
    static = static_best_arm(values)
    assert static["arm"] == 1
    assert static["improvement_pct"] == [45.0, 45.0]
    assert ttt_oracle(values)["label"].startswith("diagnostic")
