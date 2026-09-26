"""Read-only verification of the delegated approval instance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .io import read_json, sha256_file


class ApprovalError(RuntimeError):
    pass


@dataclass(frozen=True)
class ApprovalScope:
    scope_id: str
    approved: bool
    required_receipts: tuple[str, ...]
    budget_cap_card_hours: float
    mutually_exclusive_with: tuple[str, ...]


class ApprovalVerifier:
    """Verifier with no write access to approval files."""

    def __init__(self, approval_path: str | Path, sha_path: str | Path, *, expected_prompt_sha: str | None = None,
                 expected_amendments_sha: str | None = None, expected_r3_manifest_sha: str | None = None):
        self.approval_path = Path(approval_path)
        self.sha_path = Path(sha_path)
        if not self.approval_path.is_file() or not self.sha_path.is_file():
            raise ApprovalError("missing approval instance or hash file")
        self.data = read_json(self.approval_path)
        expected = self._read_hash_file()
        actual = sha256_file(self.approval_path)
        if expected != actual:
            raise ApprovalError(f"approval SHA mismatch: expected {expected}, got {actual}")
        if self.data.get("issued") is not True:
            raise ApprovalError("approval instance is not issued")
        binds = self.data.get("binds", {})
        for key, wanted in (("codex_execute_prompt_sha256", expected_prompt_sha),
                            ("required_amendments_r3_sha256", expected_amendments_sha),
                            ("r3_manifest_sha256", expected_r3_manifest_sha)):
            if wanted is not None and binds.get(key) != wanted:
                raise ApprovalError(f"approval bind mismatch: {key}")
        raw_scopes = self.data.get("scopes")
        if not isinstance(raw_scopes, list):
            raise ApprovalError("scopes must be a list")
        self.scopes = {}
        for raw in raw_scopes:
            sid = raw.get("scope_id")
            if not isinstance(sid, str) or sid in self.scopes:
                raise ApprovalError("invalid or duplicate scope id")
            self.scopes[sid] = ApprovalScope(
                sid, bool(raw.get("approved")), tuple(raw.get("requires_receipts", [])),
                float(raw.get("budget_cap_card_hours", 0)), tuple(raw.get("mutually_exclusive_with", [])),
            )

    def _read_hash_file(self) -> str:
        text = self.sha_path.read_text(encoding="utf-8").strip().split()
        if not text:
            raise ApprovalError("empty approvals.sha256")
        value = text[0].lower()
        if len(value) != 64:
            raise ApprovalError("invalid approvals.sha256 format")
        return value

    def require(self, requested: Iterable[str], *, available_receipts: Iterable[str] = ()) -> tuple[ApprovalScope, ...]:
        ids = tuple(dict.fromkeys(requested))
        unknown = [sid for sid in ids if sid not in self.scopes]
        if unknown:
            raise ApprovalError(f"unknown scope(s): {unknown}")
        for sid in ids:
            scope = self.scopes[sid]
            if not scope.approved:
                raise ApprovalError(f"scope not approved: {sid}")
        active = set(ids)
        for sid in ids:
            conflict = active.intersection(self.scopes[sid].mutually_exclusive_with)
            if conflict:
                raise ApprovalError(f"mutually exclusive scopes: {sid} with {sorted(conflict)}")
        available = set(available_receipts)
        missing = sorted({r for sid in ids for r in self.scopes[sid].required_receipts if r not in available})
        if missing:
            raise ApprovalError(f"approval receipt prerequisites missing: {missing}")
        return tuple(self.scopes[sid] for sid in ids)

    def budget_for(self, scope_id: str) -> float:
        if scope_id not in self.scopes:
            raise ApprovalError(f"unknown scope: {scope_id}")
        return self.scopes[scope_id].budget_cap_card_hours

    def snapshot(self) -> dict:
        return {"path": str(self.approval_path), "sha256": sha256_file(self.approval_path),
                "approver": self.data.get("approver"), "issued": self.data.get("issued")}
