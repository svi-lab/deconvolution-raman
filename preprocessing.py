#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:53:55 2021

@author: dejan
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from sklearn.experimental import enable_iterative_imputer
from sklearn import preprocessing, impute
import calculate as cc


def scale(spectra, **kwargs):
    """
    scale the spectra

    Parameters
    ----------
    spectra : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    return preprocessing.robust_scale(spectra, axis=-1,
                                      with_centering=False,
                                      quantile_range=(5,95))


def order(spectra, x_values):
    """
    Order values so that x_values grow.

    Parameters:
    -----------
    spectra: numpy array
        Your input spectra
    x_values: 1D numpy array
        Raman shifts

    Returns:
    --------
    ordered input values
    """

    if np.all(np.diff(x_values) <= 0):
        x_values = x_values[::-1]
        spectra = spectra[:,::-1]
    return spectra, x_values


def find_zeros(spectra):
    """
    Find the indices of zero spectra.

    Parameters
    ----------
    spectra : 2D numpy array
        your raw spectra.

    Returns
    -------
    1D numpy array of ints
        indices of zero spectra.

    """
    zero_idx = np.where((np.max(spectra, axis=-1) == 0) &
                        (np.sum(spectra, axis=-1) == 0))[0]
    if len(zero_idx) > 0:
        return zero_idx


def find_saturated(spectra, saturation_limit=90000):
    """
    Identify the saturated instances in the spectra.
    IMPORTANT: It will work only before any scaling is done!

    Parameters
    ----------
    spectra : 2D numpy array of floats
        Your input spectra.

    Returns
    -------
    Indices of saturated spectra.
    """

    razlika = np.abs(
                np.diff(spectra, n=1, axis=-1,
                        append=spectra[:,-2][:,None]))

    saturated_indices = np.unique(
                        np.where(razlika > saturation_limit)[0])

    if len(saturated_indices)==0 and np.any(spectra==0):
        print("No saturated spectra is found;\n"
              "Please make sure to apply this function before any scaling is done!")
    else:
        return saturated_indices


def get_neighbourhood(indices, map_shape):
    """
    Recover the indices of the neighbourhood (the `O`s in the schema below)
                                  O
                                 OOO
                                OOXOO
                                 OOO
                                  O
    for each element `X` listed in `indices`,
    given the shape of the containing matrix `map_shape`.
    """
    if isinstance(map_shape, int):
        nx = 1
        size = map_shape
    elif len(map_shape) == 2:
        nx = map_shape[1]
        size = map_shape[0] * map_shape[1]
    else:
        print("Check your `map_shape` value.")
        return
    extended = list(indices)
    for s in extended:
        susjedi = np.unique(
                    np.array([s-2*nx,
                              s-nx-1, s-nx, s-nx+1,
                              s-2, s-1, s, s+1, s+2,
                              s+nx-1, s+nx, s+nx+1,
                              s+2*nx]))
        susjedi_cor = susjedi[(susjedi >= 0) & (susjedi < size)]
        extended = extended + list(susjedi_cor)
    return np.sort(np.unique(extended))


def correct_zeros(rawspectra, copy=False):
    if copy:
        spectra = np.copy(rawspectra)
    else:
        spectra = rawspectra
    zero_idx = find_zeros(spectra)
    if zero_idx is not None:
        spectra[zero_idx] = np.median(spectra, axis=0)
    return spectra


def correct_saturated(rawspectra, map_shape, copy=False,
                     n_nearest_features=8, max_iter=44,
                     smoothen=True, lam=None):
    """
    Correct saturated spectra.

    Parameters:
    -----------
    rawspectra: 2D numpy array
        Your raw (!) input spectra that you want to correct
        Note that you
    map_shape: int or a tuple ints
        since this method.
    Returns
    -------
    spectra: 2D numpy array of the same shape as the input
        Your input spectra with saturated instances corrected."""

    if lam == None:
        lam = rawspectra.shape[-1]//5
    spectra = correct_zeros(rawspectra, copy=copy)
    saturated_idx = np.where(spectra==0)
# =====================this fails for very intense cosmic rays=================
#     sat = find_saturated(spectra)
#     assert(np.all(sat == np.unique(saturated_idx[0]))), "Strange saturations.\n"+\
#                 "Check if you haven't done some normalization on the spectra beforehand."
# =============================================================================
    sat = np.unique(saturated_idx[0])
    if len(sat) > 0:
        spectra[saturated_idx] = np.nan
        trt = get_neighbourhood(sat, map_shape)
        # The most important part:
        # min_value = 0.5 * np.max(rawspectra[trt], axis=-1)
        imp = impute.IterativeImputer(n_nearest_features=n_nearest_features,
                                      max_iter=max_iter, skip_complete=True)
        # create an array so that trt[vrackalica] = sat
        vrackalica = np.array([np.argwhere(trt==i)[0][0] for i in sat])
        popravljeni = imp.fit_transform(spectra[trt].T).T[vrackalica]
        spectra[sat] = popravljeni
        if smoothen:
            upeglani = cc.baseline_als(popravljeni, lam=lam, p=0.6)
            is_changed = np.diff(saturated_idx[0], prepend=sat[0])!=0
            renormalizovani = []
            i = 0
            for cond in is_changed:
                if cond:
                    i+=1
                renormalizovani.append(i)
            novi = np.copy(saturated_idx)
            novi[0] = np.array(renormalizovani)
            novi = tuple(novi)
            spectra[saturated_idx] = upeglani[novi]
    return spectra
