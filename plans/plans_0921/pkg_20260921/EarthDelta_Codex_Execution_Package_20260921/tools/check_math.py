#!/usr/bin/env python3
"""Independent synthetic algebra checks. NOT tests of the EarthDelta repository."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np


def run_checks() -> list[dict]:
    rng = np.random.default_rng(20260921)
    checks = []
    def verify(name, actual, expected, note=''):
        actual, expected = np.asarray(actual), np.asarray(expected)
        ok = bool(np.allclose(actual, expected, atol=1e-11, rtol=1e-11))
        checks.append({'name': name, 'passed': ok,
                       'max_abs_error': float(np.max(np.abs(actual-expected))), 'note': note})
    e, du = rng.normal(size=9), rng.normal(size=9)
    q = rng.uniform(.1, 2, size=9); q /= q.sum()
    norm = lambda x: np.sum(q*x*x)
    actual = norm(e)-norm(e-du)
    analytic = 2*np.sum(q*e*du)-norm(du)
    verify('weighted_finite_response_identity', analytic, actual)
    pred_minus_truth = -e
    verify('opposite_error_sign_requires_negative_cross_term',
           -2*np.sum(q*pred_minus_truth*du)-norm(du), actual)
    x, a = rng.normal(size=9), rng.normal(size=9)
    transform = lambda v: v*v + .3*v
    y0, ya, y = transform(x), transform(x+a), transform(x+e)
    te, td = y-y0, ya-y0
    verify('nonlinear_endpoint_transform_identity',
           2*np.sum(q*te*td)-norm(td), norm(te)-norm(te-td),
           'This does not imply transform(error)==error of transformed endpoints.')
    r = rng.normal(size=(9, 3)); c = rng.normal(size=3)
    h = r.T @ (q[:, None]*r); b = r.T @ (q*e)
    verify('gram_linear_surrogate', 2*b@c-c@h@c, norm(e)-norm(e-r@c))
    first, second = r[:, 0], r[:, 1]
    verify('linear_cross_term', norm(first+second)-norm(first)-norm(second),
           2*np.sum(q*first*second))
    nonlinear_response = lambda v: r@v + .7*v[0]*v[1]*np.ones(9)
    u, v = np.array([.2,0,0]), np.array([0,.4,0])
    mixed = nonlinear_response(u+v)-nonlinear_response(u)-nonlinear_response(v)
    verify('nonlinear_interaction_separate_from_gram_overlap', mixed, .7*.2*.4*np.ones(9))
    matrix = np.array([[.9,.3],[-.1,1.1]])
    edit_matrix = np.array([[.01,.05],[.03,-.02]])
    state = np.array([.3,-.8]); state_edit=state.copy(); state_feedback=state.copy()
    for _ in range(5):
        state_edit=(matrix+edit_matrix)@state_edit
        state_feedback=matrix@state_feedback+edit_matrix@state_feedback
    verify('feedback_output_can_represent_parameter_dynamics', state_feedback, state_edit)
    noise = np.array([-1.,1.])
    # e=du=noise, both means are zero: plug-in mean predicts 0, true gain is 1.
    true_expected = np.mean(2*noise*noise-noise*noise)
    corrected = 0 + 2*np.mean(noise*noise)-np.mean(noise*noise)
    verify('conditional_covariance_correction', corrected, true_expected,
           'Relevant with compressed information; exact deterministic du has no conditional variance given full input.')
    ep, dp = rng.normal(size=9), rng.normal(size=9)
    new_gain=2*np.sum(q*(e+ep)*(du+dp))-norm(du+dp)
    expansion=(2*np.sum(q*ep*du)+2*np.sum(q*(e-du)*dp)
               +2*np.sum(q*ep*dp)-norm(dp))
    verify('gain_prediction_error_expansion', new_gain-analytic, expansion)
    verify('no_edit_gain_is_zero', norm(e)-norm(e), 0.)
    return checks


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error('Output exists; no overwrite allowed.')
    checks=run_checks()
    result={'created_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'independent synthetic algebra only; no repository code imported',
            'repository_tests_run':False, 'weather_experiments_run':False,
            'checks':checks,'passed':sum(x['passed'] for x in checks),'total':len(checks)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(f"Synthetic algebra: {result['passed']}/{result['total']}; not repository or weather validation.")
    return 0 if result['passed']==result['total'] else 1

if __name__=='__main__':
    raise SystemExit(main())
