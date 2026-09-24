"""WeatherBench-X / WeatherBench 2 evaluation bridge for EarthDelta.

EarthDelta does not implement its own standard deterministic weather metrics.
This subpackage exports EarthDelta forecasts (Stormer F0, certified Fs, dynamic
policy) into the official WeatherBench forecast layout and then scores them
with the vendored, pinned official code:

* ``reference/weatherbenchX`` -- metrics (RMSE/MSE/Bias/ACC/WindVectorRMSE),
  area weighting, region binning, chunked aggregation, xarray data loaders.
* ``reference/weatherbench2`` -- the official conservative regridder and the
  forecast time-naming convention (``schema.apply_time_conventions``).

Only the apache_beam-free parts of both repos are used; ``weatherbench2.
evaluation`` / ``weatherbench2.metrics`` and ``weatherbenchX.beam_pipeline`` are
never imported (``_vendor.assert_no_beam`` enforces this).

Modules
-------
``_vendor``   sys.path insertion + fail-closed pin verification against
              ``reference/_manifest.json``; optional ``.pydeps_wbx`` (CPU jax).
``export``    69-channel <-> (variable, level) conversion, denormalization of
              normalized rollouts, official ``time``/``prediction_timedelta``
              forecast datasets, atomic zarr publish, truth extraction.
``regrid``    official WB2 conservative regridding (comparison-only; Stormer's
              native 128x256 rollout is never regridded) plus cross-checks.
``evaluate``  WB-X metric driver without beam, and the evaluation-year gate.
``baselines`` official public baseline store registry (stub for the smoke test).

EarthDelta-specific intervention metrics (oracle gain, policy gain, regret,
harmful-edit rate, no-edit rate) are NOT computed here; they stay in
``earthdelta.metrics_contract`` and the FP-05 candidate-cache code.

Nothing here is imported by the pinned Fs/bank code, and importing
``earthdelta.wbx`` itself pulls in no third-party dependency: every submodule
imports xarray/jax lazily through ``_vendor``.
"""

import os as _os
import sys as _sys

#: Container quirk (README "Quickstart"): the system onnx package's generated
#: ``*_pb2.py`` files cannot be parsed by the upb protobuf backend in
#: ``.pydeps``, so ``import timm`` (-> torchvision -> onnx) raises
#: "Descriptors cannot be created directly" unless the pure-python backend is
#: selected BEFORE google.protobuf is first imported. jax/xarray/WB-X import no
#: protobuf at all; this is set here so that any process importing
#: ``earthdelta.wbx`` before torch/timm works regardless of how it was
#: launched. It is the same setting every pinned GPU job already exports.
PROTOBUF_ENV = "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"


def ensure_protobuf_python_impl() -> str:
    """Select the pure-python protobuf backend if protobuf is not loaded yet.

    Returns the implementation in effect ("python", or the already-loaded
    backend name when protobuf was imported earlier in this process).
    """
    if "google.protobuf" not in _sys.modules:
        _os.environ.setdefault(PROTOBUF_ENV, "python")
        return _os.environ[PROTOBUF_ENV]
    from google.protobuf.internal import api_implementation
    return api_implementation.Type()


ensure_protobuf_python_impl()

#: Name under which EarthDelta's Stormer forecasts are exported. EarthDelta
#: rolls out ONLY the 6h path of the ps4 checkpoint, whereas the published
#: Stormer inference ensembles the 6/12/24h paths, so the export must never be
#: labelled plain "Stormer" (that would invite a misleading comparison with the
#: paper's numbers).
STORMER_EXPORT_PREFIX = "Stormer-ps4-6h-path"

__all__ = ["STORMER_EXPORT_PREFIX", "PROTOBUF_ENV", "ensure_protobuf_python_impl"]
