#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:50:13 2021

@author: dejan
"""
import numpy as np
from scipy import sparse, ndimage
from scipy.optimize import minimize_scalar
from joblib import delayed, Parallel
from warnings import warn
from tqdm import tqdm


def find_barycentre(x: np.array, y: np.array, method: str = 'simple vectorized') -> (np.array, np.array):
    """Calculate the coordinates of the barycentre value.

    Parameters
    ----------
    x : np.array
        ndarray containing your raman shifts
    y : np.array
        ndarray containing your intensity (counts) values
    method : string
        one of ['trapz_minimize', 'list_minimize', 'weighted_mean', 'simple vectorized']

    Returns
    -------
    (x_value, y_value): the coordinates of the barycentre
    """
    assert (method in ['trapz_minimize', 'list_minimize', 'weighted_mean', 'simple vectorized'])
    if x[0] == x[-1]:
        return x * np.ones(len(y)), y / 2
    if method == 'trapz_minimize':
        half = np.abs(np.trapz(y, x=x) / 2)

        def find_y(y0, xx=x, yy=y):
            """Internal function to minimize
            depending on the method chosen"""
            # Calculate the area of the curve above the y0 value:
            part_up = np.abs(np.trapz(
                yy[yy >= y0] - y0,
                x=xx[yy >= y0]))
            # Calculate the area below y0:
            part_down = np.abs(np.trapz(
                yy[yy <= y0],
                x=xx[yy <= y0]))
            # for the two parts to be the same
            to_minimize_ud = np.abs(part_up - part_down)
            # fto make the other part be close to half
            to_minimize_uh = np.abs(part_up - half)
            # to make the other part be close to half
            to_minimize_dh = np.abs(part_down - half)
            return to_minimize_ud ** 2 + to_minimize_uh + to_minimize_dh

        def find_x(x0, xx=x, yy=y):
            part_left = np.abs(np.trapz(
                yy[xx <= x0],
                x=xx[xx <= x0]))
            part_right = np.abs(np.trapz(yy[xx >= x0],
                                         x=xx[xx >= x0]))
            to_minimize_lr = np.abs(part_left - part_right)
            to_minimize_lh = np.abs(part_left - half)
            to_minimize_rh = np.abs(part_right - half)
            return to_minimize_lr ** 2 + to_minimize_lh + to_minimize_rh

        minimized_y = minimize_scalar(find_y, method='Bounded',
                                      bounds=(np.quantile(y, 0.01),
                                              np.quantile(y, 0.99)))
        minimized_x = minimize_scalar(find_x, method='Bounded',
                                      bounds=(np.quantile(x, 0.01),
                                              np.quantile(x, 0.99)))
        y_value = minimized_y.x
        x_value = minimized_x.x

    elif method == "list_minimize":
        yy = y
        xx = x
        ys = np.sort(yy)
        z2 = np.asarray(
            [np.abs(np.trapz(yy[yy <= y_val], x=xx[yy <= y_val]) - \
                    np.trapz(yy[yy >= y_val] - y_val, x=xx[yy >= y_val])) \
             for y_val in ys])
        y_value = ys[np.argmin(z2)]
        x_ind = np.argmin(np.abs(np.cumsum(yy) - np.sum(yy) / 2)) + 1
        x_value = xx[x_ind]

    elif method == 'weighted_mean':
        weighted_sum = np.dot(y, x)
        x_value = weighted_sum / np.sum(y, axis=-1)
        y_value = weighted_sum / np.sum(x)

    elif method == 'simple vectorized':
        xgrad = np.gradient(x)
        proizvod = y * xgrad
        sumprod = np.cumsum(proizvod, axis=-1)
        medo = np.median(sumprod, axis=-1, keepdims=True)  # this should be half area
        ind2 = np.argmin(np.abs(sumprod - medo), axis=-1)
        x_value = x[ind2]
        y_value = sumprod[:, -1] / (x[-1] - x[0])
    return x_value, y_value


def rolling_median(arr, w_size, ax=0, mode='nearest', *args):
    """Calculates the rolling median of an array
    along the given axis on the given window size.
    Parameters:
    -------------
        arr:ndarray: input array
        w_size:int: the window size
                    (should be less then the dimension along the given axis)
        ax:int: the axis along which to calculate the rolling median
        mode:str: to choose from ['reflect', 'constant', 'nearest', 'mirror', 'wrap']
        see the docstring of ndimage.median_filter for details
    Returns:
    ------------
        ndarray of same shape as the input array"""
    shape = np.ones(np.ndim(arr), dtype=int)
    shape[ax] = w_size
    return ndimage.median_filter(arr, size=shape, mode=mode, *args)


def baseline_als(y, lam=1e5, p=5e-5, niter=12):
    """Adapted from:
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library.

    To get the feel on how the algorithm works, you can think of it as
    if the rolling ball which comes from beneath the spectrum and thus sets
    the baseline.

    Then, to follow the image, schematic explanation of the params would be:

    Params:
    ----------
        y:          1D or 2D ndarray: the spectra on which to find the baseline

        lam:number: Can be viewed as the radius of the ball.
                    As a rule of thumb, this value should be around the
                    twice the width of the broadest feature you want to keep
                    (width is to be measured in number of points, since
                    for the moment no x values are taken into account
                    in this algorithm)

        p:number:   Can be viewed as the measure of how much the ball
                    can penetrate into the spectra from below

        niter:int:  number of iterations
                   (the resulting baseline should stabilize after
                    some number of iterations)

    Returns:
    -----------
        b_line:ndarray: the baseline (same shape as y)

    Note:
    ----------
        It takes around 2-3 sec per 1000 spectra with 10 iterations
        on i7 4cores(8threads) @1,9GHz

    """

    def _one_bl(yi, lam=lam, p=p, niter=niter, z=None):
        if z is None:
            L = yi.shape[-1]
            D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
            D = lam * D.dot(D.transpose())  # Precompute this term since it does not depend on `w`
            w = np.ones(L)
            W = sparse.spdiags(w, 0, L, L)
        for i in range(niter):
            W.setdiag(w)  # Do not create a new matrix, just update diagonal values
            Z = W + D
            z = sparse.linalg.spsolve(Z, w * yi)
            w = p * (yi > z) + (1 - p) * (yi < z)
        return z

    if y.ndim == 1:
        b_line = _one_bl(y)
    elif y.ndim == 2:
        b_line = np.asarray(Parallel(n_jobs=-1)(delayed(_one_bl)(y[i])
                                                for i in tqdm(range(y.shape[0]))))
    else:
        warn("This only works for 1D or 2D arrays")

    return b_line
