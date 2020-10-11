"""
Functions for evaluating forecasts.
"""
import numpy as np
import xarray as xr

def compute_weighted_rmse(da_fc, da_true, mean_dims="pressure_level, latitude, longitude"):
    """
    Compute the RMSE with latitudeitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        rmse: Latitude weighted root mean squared error
    """
    error = da_fc - da_true
    weights_latitude = np.cos(np.deg2rad(error.latitude))
    weights_latitude /= weights_latitude.mean()
    rmse = np.sqrt(((error)**2 * weights_latitude).mean(mean_dims))
    return rmse

def compute_weighted_acc(da_fc, da_true, mean_dims="pressure_level, latitude, longitude"):
    """
    Compute the ACC with latitudeitude weighting from two xr.DataArrays.
    WARNING: Does not work if datasets contain NaNs
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        acc: Latitude weighted acc
    """

    clim = da_true.mean('time')
    try:
        t = np.intersect1d(da_fc.da_true.time)
        fa = da_fc.sel(time=t) - clim
    except AttributeError:
        t = da_true.time.values
        fa = da_fc - clim
    a = da_true.sel(time=t) - clim

    weights_latitude = np.cos(np.deg2rad(da_fc.latitude))
    weights_latitude /= weights_latitude.mean()
    w = weights_latitude

    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()

    acc = (
            np.sum(w * fa_prime * a_prime) /
            np.sqrt(
                np.sum(w * fa_prime ** 2) * np.sum(w * a_prime ** 2)
            )
    )
    return acc

def compute_weighted_mae(da_fc, da_true, mean_dims="pressure_level, latitude, longitude"):
    """
    Compute the MAE with latitudeitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        mae: Latitude weighted root mean absolute error
    """
    error = da_fc - da_true
    weights_latitude = np.cos(np.deg2rad(error.latitude))
    weights_latitude /= weights_latitude.mean()
    mae = (np.abs(error) * weights_latitude).mean(mean_dims)
    return mae
