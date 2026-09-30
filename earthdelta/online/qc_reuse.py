"""Content-bound partial QC reuse. No scanning or downloads on import."""
from pathlib import Path
import numpy as np
from .io import checked_json, sha256_file, decoded_sha256
from .clock import immutable_array
from .timeindex import checked_index


def verify_qc_sources(receipt_path, receipt_sha256, spec_path, spec_sha256,
                      *, store, metadata_bindings):
    receipt = checked_json(receipt_path, receipt_sha256)
    spec = checked_json(spec_path, spec_sha256)
    if receipt.get('status') != 'OBSERVED' or receipt.get('spec_sha256') != spec_sha256:
        raise ValueError('old QC is not bound to an observed specification')
    if Path(receipt['store']).resolve() != Path(store).resolve():
        raise ValueError('QC store identity mismatch')
    for relative, expected in metadata_bindings.items():
        if sha256_file(Path(store)/relative) != expected:
            raise ValueError('store metadata identity changed')
    if metadata_bindings.get('COMPLETE.json') != receipt['metadata']['COMPLETE_sha256']:
        raise ValueError('COMPLETE identity not bound to prior QC')
    return receipt, spec


def compose_index_manifest(old_receipt, new_rows, *, support, missing, overlap,
                           normalization_identity, z_threshold=40.):
    support = tuple(checked_index(i) for i in support)
    missing = set(missing)
    if len(set(support)) != len(support) or not missing.issubset(support):
        raise ValueError('invalid registered support')
    if not normalization_identity or not np.isfinite(z_threshold) or z_threshold <= 0:
        raise ValueError('bound normalization and QC rule required')
    needed = set(support)-missing
    old = old_receipt['per_index']
    new_only = {i for i in needed if str(i) not in old}
    if set(new_rows) != new_only | set(overlap) or not set(overlap).issubset(needed-new_only):
        raise ValueError('new QC cannot shrink or expand the registered 17+8 scope')
    for i in overlap:
        if new_rows[i]['qc']['decoded_sha256'] != old[str(i)]['decoded_sha256']:
            raise ValueError('overlap content changed')
    result = {}
    for i in sorted(needed):
        if i in new_only:
            row = dict(new_rows[i])
            row['origin'] = 'new_partial_qc'
        else:
            row = {'qc':dict(old[str(i)]), 'cache':dict(old_receipt['cache_files'][str(i)]),
                   'origin':'bound_old_cache'}
        qc = row['qc']
        if qc['index'] != i or not isinstance(qc['finite'],bool):
            raise ValueError('QC index/type mismatch')
        row['eligible'] = bool(qc['finite'] and np.isfinite(qc['max_abs_z']) and qc['max_abs_z']<=z_threshold)
        result[str(i)] = row
    return {'schema':'earthdelta.online.partial_qc.v1','scope':'registered_indices_only',
            'whole_year_certified':False,'normalization_identity':normalization_identity,
            'z_threshold':z_threshold,'missing_indices':sorted(missing),'indices':result}


def load_bound_cache(row):
    """Verify bytes before decoding, then shape/dtype/decoded identity every open."""
    if not row['eligible']:
        return None
    info = row['cache']
    path = Path(info['path'])
    # Same open descriptor prevents a path substitution between hash and load.
    import hashlib
    with path.open('rb') as f:
        h=hashlib.sha256()
        for part in iter(lambda:f.read(8*1024*1024), b''): h.update(part)
        if h.hexdigest()!=info['sha256']: raise ValueError('cache byte identity changed')
        f.seek(0); x=np.load(f,allow_pickle=False)
    if list(x.shape)!=info['shape'] or str(x.dtype)!=info['dtype']:
        raise ValueError('cache array layout mismatch')
    if decoded_sha256(x)!=row['qc']['decoded_sha256'] or not np.isfinite(x).all():
        raise ValueError('cache decoded identity changed')
    return immutable_array(x)
