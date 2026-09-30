"""Physical area MSE and differentiable normalization with explicit dimensions."""
import numpy as np
import torch
from numbers import Integral
from earthdelta.v8.standard_metrics import issue_mse


def area_weights(latitude, width, *, dtype=torch.float64, device=None):
    lat = torch.as_tensor(latitude, dtype=dtype, device=device)
    if lat.ndim != 1 or not len(lat) or isinstance(width, bool) or not isinstance(width, Integral) or width <= 0 or not bool(torch.isfinite(lat).all()) or bool((lat.abs() >= 90).any()):
        raise ValueError('regular global cell-center latitudes required')
    w = torch.cos(torch.deg2rad(lat))[:, None].expand(-1, int(width))
    return w / w.sum()


def normalized_step_mse(prediction, truth, latitude, denominator):
    """[batch, step, variable, height, width], with physical-MSE [step,variable] D."""
    if prediction.shape != truth.shape or prediction.ndim != 5:
        raise ValueError('matching [B,T,C,H,W] fields required')
    d = torch.as_tensor(denominator, dtype=prediction.dtype, device=prediction.device)
    if tuple(d.shape) != tuple(prediction.shape[1:3]) or not bool(torch.isfinite(d).all()) or bool((d <= 0).any()):
        raise ValueError('positive finite physical-MSE denominator required')
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(truth).all()):
        raise ValueError('nonfinite field')
    w = area_weights(latitude, prediction.shape[-1], dtype=prediction.dtype, device=prediction.device)
    if w.shape != prediction.shape[-2:]:
        raise ValueError('latitude count does not identify the field grid')
    return (((prediction-truth).square()*w).sum((-2,-1))/d).mean()


def cells(prediction, truth, latitude):
    """Physical MSE [issue,variable,lead], using the frozen NumPy scorer."""
    # Cast BEFORE subtraction/squaring. Otherwise a float32 cache and an
    # equivalent float64 view produce different reported cells.
    p,y = np.asarray(prediction,dtype=np.float64),np.asarray(truth,dtype=np.float64)
    if p.shape != y.shape or p.ndim != 5:
        raise ValueError('matching [issue,lead,variable,height,width] required')
    return np.stack([issue_mse(p[:,:,v], y[:,:,v], latitude) for v in range(p.shape[2])],axis=1)


def validate_wbx_binding(forecast, truth, init_times, valid_times):
    """Bind caller-supplied time arguments to the actual xarray coordinates."""
    if valid_times is None or init_times is None or set(forecast) != set(truth):
        raise ValueError('valid-time identity required')
    for key,p in forecast.items():
        y=truth[key]
        if p.dims != y.dims or p.shape != y.shape:
            raise ValueError('WBX field layout mismatch')
        for name in ('init_time','lead_time','latitude','longitude'):
            if name not in p.coords or name not in y.coords or not np.array_equal(p[name].values,y[name].values):
                raise ValueError('WBX coordinate binding mismatch')
        inits=p['init_time'].values.astype('datetime64[ns]')
        leads=p['lead_time'].values.astype('timedelta64[ns]')
        expected=(inits[:,None]+leads[None,:]).ravel()
        if not np.array_equal(np.asarray(init_times).astype('datetime64[ns]'),inits) or not np.array_equal(np.asarray(valid_times).astype('datetime64[ns]').ravel(),expected):
            raise ValueError('WBX time arguments do not identify these arrays')
        if np.any(expected.astype('datetime64[Y]') != np.datetime64('2020','Y')):
            raise PermissionError('online stage 0 permits only 2020 valid times')


def wbx_evaluate(forecast, truth, *, init_times, valid_times):
    validate_wbx_binding(forecast,truth,init_times,valid_times)
    from earthdelta.wbx.evaluate import evaluate_single_chunk, standard_metrics
    return evaluate_single_chunk(forecast,truth,metrics=standard_metrics(wind_vector=False),init_times=init_times,valid_times=valid_times)
