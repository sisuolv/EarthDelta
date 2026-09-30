import json
import os
from pathlib import Path
import numpy as np
import pytest
from earthdelta.online.cost import projected_card_hours


def test_cost_counts_all_nominal_work_and_failed_jobs():
    contract=json.loads(Path(os.environ['ONLINE_CONTRACT']).read_text())
    t={'roll20':10.,'roll4':2.,'grad4':8.,'grad2':4.}
    out=projected_card_hours(t,contract,charged_fit_hours=.5)
    expected=(86*10+86*8+8*16*14*2+24*26*10)*1.2/3600+.5
    assert out['fit_card_hours_upper_projection']==pytest.approx(expected)
    assert not out['admitted']
    fast=projected_card_hours({k:.1 for k in t},contract,charged_fit_hours=.5)
    assert fast['admitted']
    with pytest.raises(ValueError):projected_card_hours(dict(t,roll20=np.nan),contract,charged_fit_hours=0)
