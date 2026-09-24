#!/usr/bin/env python3
"""Recompute arithmetic from transcribed evidence, not a weather experiment."""
import argparse
import datetime as dt
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    source=json.loads((a.package/'evidence/INDEPENDENT_ARITHMETIC.json').read_text())
    obs=json.loads((a.package/'evidence/OBSERVATIONS.json').read_text())
    idx=source['x1_index_list'];fs=obs['fs_v2'];base=dt.datetime(2020,1,1,tzinfo=dt.timezone.utc)
    out={'state':'OBSERVED','scope':'SCALAR_ARITHMETIC_NOT_NEW_MODEL_RESULTS','n_x1_extra':len(idx),
         'largest_gap_hours':max((b-a)*6 for a,b in zip(idx,idx[1:])),
         'support_start':(base+dt.timedelta(hours=idx[0]*6-12)).isoformat(),
         'support_end':(base+dt.timedelta(hours=idx[-1]*6+72)).isoformat(),
         'support_union_contiguous':all((b-a)*6<=84 for a,b in zip(idx,idx[1:])),
         'train24_ratio_from_means':fs['train24_Fs_mean']/fs['train24_F0_mean'],
         'holdout24_ratio_from_means':fs['holdout24_Fs_mean']/fs['holdout24_F0_mean'],
         'same_panel_break_even_Fs_fraction':(fs['holdout24_Fs_mean']-fs['holdout24_F0_mean'])/fs['holdout24_Fs_mean']}
    out['checks']={'n56':len(idx)==56,'support_connected':out['support_union_contiguous'],
                   'train_ratio':abs(out['train24_ratio_from_means']-fs['train24_ratio'])<1e-14,
                   'holdout_ratio':abs(out['holdout24_ratio_from_means']-fs['holdout24_ratio'])<1e-14,
                   'denominator_conversion':abs(out['same_panel_break_even_Fs_fraction']-source['same_panel_break_even_fraction_of_Fs'])<1e-14}
    if a.out.exists():p.error('output exists; use a new file')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out,indent=2));return 0 if all(out['checks'].values()) else 1
if __name__=='__main__':raise SystemExit(main())
