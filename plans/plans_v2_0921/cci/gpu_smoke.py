#!/usr/bin/env python3
"""4-GPU ACP smoke probe. Run in an ACP worker using the adjacent request.json;
writes one result.json to AFS. Checks, beyond the 1-GPU probe:

1. torch.cuda.device_count() == expected_gpu_count (primary check).
2. Per-device: allocate a tensor, matmul, compare against a CPU reference,
   independently on every visible device.
3. A single-node multi-GPU NCCL all_reduce collective across all visible
   devices via torch.multiprocessing.spawn, asserting every rank gets the
   correct summed result.
4. An AFS read/write check: write a marker file into the run directory (on
   the shared AFS mount) and read it back to confirm the round trip.
"""

import datetime
import hashlib
import json
import os
import socket
import sys
import traceback
from pathlib import Path


def _nccl_worker(rank, world_size, master_port, out_dir):
    # Runs in a spawned subprocess (fresh interpreter, "spawn" start method).
    # Writes its own tiny JSON result file instead of returning a value,
    # since torch.multiprocessing.spawn does not collect return values.
    result = {'rank': rank, 'ok': False}
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(master_port)
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend='nccl', init_method='env://',
            rank=rank, world_size=world_size,
            timeout=datetime.timedelta(seconds=60),
        )
        value = float(rank + 1)
        tensor = torch.tensor([value], dtype=torch.float32, device='cuda:%d' % rank)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        got = tensor.item()
        expected = float(sum(range(1, world_size + 1)))
        result['input_value'] = value
        result['all_reduce_sum'] = got
        result['expected_sum'] = expected
        result['ok'] = abs(got - expected) < 1e-4
        dist.barrier()
        dist.destroy_process_group()
    except Exception as exc:
        result['error'] = '%s: %s' % (type(exc).__name__, exc)
        result['traceback'] = traceback.format_exc()
    (Path(out_dir) / ('nccl_rank_%d.json' % rank)).write_text(json.dumps(result) + '\n')


def _free_tcp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    directory = Path(__file__).resolve().parent
    request_bytes = (directory / 'request.json').read_bytes()
    request = json.loads(request_bytes)
    output = directory / 'result.json'
    if output.exists():
        raise RuntimeError('A result already exists; use a new run directory instead of overwriting it.')

    result = {
        'status': 'failed',
        'run_id': request['run_id'],
        'expected_gpu': request['expected_gpu'],
        'expected_gpu_count': request.get('expected_gpu_count'),
        'afs_marker': request['afs_marker'],
        'request_sha256': hashlib.sha256(request_bytes).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'hostname': socket.gethostname(),
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    try:
        if result['hostname'] == request['source_cci_hostname']:
            raise RuntimeError('This probe must run in an ACP worker, not in the source CCI.')

        import torch

        result['torch_version'] = str(torch.__version__)
        result['cuda_version'] = torch.version.cuda
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this worker; check its resource specification and image.')

        count = torch.cuda.device_count()
        result['visible_gpu_count'] = count
        expected_count = int(request.get('expected_gpu_count', 4))
        if count != expected_count:
            raise RuntimeError(
                'PRIMARY CHECK FAILED: expected exactly %d visible GPUs, found %d.' % (expected_count, count)
            )

        # --- per-device identity + matmul checks (independent per device) ---
        per_device = []
        torch.backends.cuda.matmul.allow_tf32 = False
        expected_gpu = request['expected_gpu']
        for idx in range(count):
            props = torch.cuda.get_device_properties(idx)
            entry = {
                'index': idx,
                'gpu_name': props.name,
                'memory_gib': round(props.total_memory / (1024 ** 3), 3),
                'compute_capability': list(torch.cuda.get_device_capability(idx)),
            }
            if expected_gpu == 'H100':
                matches = 'H100' in props.name.upper()
            elif expected_gpu == 'RTX 5090':
                matches = '5090' in props.name.upper() and 'RTX' in props.name.upper()
            else:
                raise RuntimeError('Unsupported requested GPU: %s' % expected_gpu)
            entry['gpu_name_matches_expected'] = matches
            if not matches:
                raise RuntimeError('Device %d: requested %s, received %s.' % (idx, expected_gpu, props.name))

            # Distinct-but-deterministic source matrix per device so each
            # device check is independent, not a repeat of the same math.
            base = torch.arange(256 * 256, dtype=torch.float32).reshape(256, 256) / (256 * 256)
            source = base + float(idx)
            expected_result = source @ source.T
            operand = source.to('cuda:%d' % idx)
            actual = operand @ operand.T
            torch.cuda.synchronize()
            copied = actual.cpu()
            max_err = float((copied - expected_result).abs().max())
            matches_cpu = bool(torch.allclose(copied, expected_result, rtol=1e-4, atol=1e-4))
            entry['max_absolute_error'] = max_err
            entry['result_matches_cpu'] = matches_cpu
            per_device.append(entry)
            if not matches_cpu:
                raise RuntimeError('Device %d GPU matmul result did not match the CPU reference.' % idx)

        result['per_device'] = per_device
        result['all_devices_match_cpu'] = all(d['result_matches_cpu'] for d in per_device)

        # --- single-node multi-GPU NCCL all_reduce collective ---
        import torch.multiprocessing as mp

        world_size = count
        master_port = _free_tcp_port()
        nccl_dir = directory / 'nccl_tmp'
        nccl_dir.mkdir(exist_ok=True)
        mp.spawn(_nccl_worker, args=(world_size, master_port, str(nccl_dir)), nprocs=world_size, join=True)

        nccl_results = []
        for rank in range(world_size):
            rank_file = nccl_dir / ('nccl_rank_%d.json' % rank)
            if not rank_file.exists():
                raise RuntimeError('NCCL rank %d produced no result file; collective likely failed or hung.' % rank)
            nccl_results.append(json.loads(rank_file.read_text()))

        result['nccl_all_reduce'] = nccl_results
        result['nccl_world_size'] = world_size
        result['nccl_all_ranks_ok'] = bool(
            len(nccl_results) == world_size and all(r.get('ok') for r in nccl_results)
        )
        if not result['nccl_all_ranks_ok']:
            raise RuntimeError('NCCL all_reduce did not produce the correct summed result on every rank.')

        # --- AFS read/write check (round trip through the shared mount) ---
        afs_check_path = directory / 'afs_write_check.txt'
        afs_payload = 'afs_marker=%s hostname=%s\n' % (request['afs_marker'], result['hostname'])
        afs_check_path.write_text(afs_payload)
        read_back = afs_check_path.read_text()
        result['afs_write_check'] = {
            'path': str(afs_check_path),
            'written': afs_payload,
            'read_back_matches': read_back == afs_payload,
        }
        if not result['afs_write_check']['read_back_matches']:
            raise RuntimeError('AFS read/write check failed: read-back content did not match what was written.')

        result['status'] = 'ok'
    except Exception as exc:
        result['error'] = '%s: %s' % (type(exc).__name__, exc)
        result['traceback'] = traceback.format_exc()
    result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    serialized = json.dumps(result, indent=2) + '\n'
    with output.open('x', encoding='utf-8') as stream:
        stream.write(serialized)
    print(serialized, end='', flush=True)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
