"""Fail-closed execution primitives for the EarthDelta v8 contract.

The v8 namespace is deliberately isolated from the frozen research code.  The
modules here validate contracts, receipts, data roles and metrics; expensive
model execution is delegated to the phase runner and is never represented by a
synthetic success receipt.
"""

from .approvals import ApprovalError, ApprovalVerifier
from .receipt_registry import ReceiptError, ReceiptRegistry
from .standard_metrics import MetricContract, pooled_rmse, paired_bootstrap

__all__ = [
    "ApprovalError",
    "ApprovalVerifier",
    "ReceiptError",
    "ReceiptRegistry",
    "MetricContract",
    "pooled_rmse",
    "paired_bootstrap",
]
