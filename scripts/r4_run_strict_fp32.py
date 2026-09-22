#!/usr/bin/env python3
"""Use one declared FP32 policy for official S0 and the pilot's model runs."""
import json
import runpy
import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print(json.dumps({"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                  "torch": str(torch.__version__), "entry": sys.argv[1]}), flush=True)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
