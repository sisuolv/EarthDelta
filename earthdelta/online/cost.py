"""Conservative resource admission from measured per-issue times."""
import numpy as np


def projected_card_hours(timing,contract,*,charged_fit_hours,charged_analysis_hours=0.):
    values={k:float(timing[k]) for k in ('roll20','roll4','grad4','grad2')}
    if not np.isfinite(list(values.values())+[charged_fit_hours,charged_analysis_hours]).all() or min(values.values())<=0 or min(charged_fit_hours,charged_analysis_hours)<0:
        raise ValueError('finite positive timings and nonnegative ledger charges required')
    n=contract['budget']['nominal_counts'];t=values
    # Already allocated preflight/failed jobs are charged separately; the
    # remaining nominal work is not discounted for missing cases or parallelism.
    a3=(86*t['roll20']+86*t['grad4']+n['calibration_variants']*n['calibration_issues']*n['calibration_calls_max']*t['roll4']+n['selection_configs']*n['selection_issues']*t['roll20'])*1.2/3600+charged_fit_hours
    a4=(275*t['roll20']+275*t['grad4']+n['analysis_arms']*n['analysis_issues']*t['roll20']+n['diagnostic_issues']*(n['diagnostic_rollouts_each']*t['roll20']+t['grad2']))*1.2/3600+charged_analysis_hours
    budget=contract['budget']
    ok=a3<=budget['proposed_A3_fit_cap'] and a4<=budget['proposed_A4_analysis_cap'] and a3+a4<=budget['proposed_total_cap']
    return {'fit_card_hours_upper_projection':a3,'analysis_card_hours_upper_projection':a4,
            'total_card_hours_upper_projection':a3+a4,'admitted':bool(ok),
            'basis':'p95 per-issue timing, full nominal counts, 20% future overhead, plus already charged jobs',
            'not_a_measured_total':True}
