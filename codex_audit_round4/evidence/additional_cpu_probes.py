"""Small CPU counterexamples and source comparisons for the Round-4 audit."""
import ast
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

from earthdelta.contracts import EditPlan
from earthdelta.metrics_contract import WeightConvention, quadratic_gain
from earthdelta.registry import CandidateRegistry, RegistryViolation
from earthdelta.selection import select_from_registry


ROOT = Path(__file__).resolve().parents[2]
results = {}

error = torch.ones(1, 2, 1, 2)
response = torch.ones(1, 1, 2, 1, 2)
weights = torch.tensor([[[[1., 1.]], [[0., 0.]]]])
gain = quadratic_gain(error, response, weights, convention=WeightConvention.WEIGHTED_MEAN)
assert torch.isnan(gain[0, 0, 1, 0])
results['legacy_zero_slice'] = {
    'gain_repr': str(gain.tolist()),
    'nonfinite_count': int((~torch.isfinite(gain)).sum()),
}

registry = CandidateRegistry(num_experts=1)
try:
    registry.register(SimpleNamespace(plan_id='complex', coefficients=(0.1 + 100j,),
                                      num_experts=1, rho=0.25))
except RegistryViolation as exc:
    results['registry_complex_direct_rejection'] = {'code': exc.code, 'message': str(exc)}
else:
    raise AssertionError('registry accepted complex coefficients')

registry.register_reference(1)
registry.register(EditPlan('small_positive', 1, (5e-7,), rho=0.25), source='audit fixture')
selection = select_from_registry(registry, torch.tensor([1.]), torch.eye(1), ridge=0., max_active=1)
assert selection.is_reference
assert float(selection.surrogate.coefficients[0]) > 0.
results['registry_ambiguous_coefficient_match'] = {
    'selected_plan_id': selection.plan_id,
    'reported_coefficients': selection.coefficients,
    'actual_planner_coefficients': selection.surrogate.coefficients.tolist(),
    'planner_predicted_gain': selection.surrogate.predicted_gain,
}

path = 'earthdelta/selection.py'
before = subprocess.check_output(['git', '--git-dir=.git', '--work-tree=.', 'show', 'HEAD:' + path],
                                 cwd=ROOT, text=True)
after = (ROOT / path).read_text()
names = ('unified_select', 'select_plan', 'plan_from_prediction', '_plan_from_finite_candidates')
def functions(source):
    return {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef)}
old, new = functions(before), functions(after)
results['existing_selection_functions_source_unchanged'] = {name: old[name] == new[name] for name in names}
assert all(results['existing_selection_functions_source_unchanged'].values())

output = ROOT / 'codex_audit_round4/evidence/additional_cpu_probes.json'
output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
print(output.read_text())
