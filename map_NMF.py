# -*- coding: utf-8 -*-
# %%
import os
import numpy as np
import pandas as pd
from sklearn import decomposition
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib import colors
import seaborn as sns
from tkinter import filedialog, Tk, messagebox
from timeit import default_timer as time
from read_WDF import convert_time, read_WDF
from warnings import warn
import utilities as ut
import preprocessing as pp


sns.set()
'''This script uses NMF deconvolution
to produce some informative graphical output on map scans.
ATTENTION: you are supposed to provide your spectra in .wdf files)
All of the imported scripts should be in your working directory

You should first choose the data file with the map scan in the .wdf format

Set the initialization dictionary values
That'_s it!

'''
# %%
# -----------------------Choose a file-----------------------------------------

folder_name = "../../RamanData/Chloe/"
# folder_name = "./Data/Giuseppe/"
# file_n = "cBN20-532streamline-x20-2s-carto1.wdf"
# file_n = "TFCD_ITOcell_532nm_p100_1s_carto_z20.wdf"
file_n = "LFeige-532streamline-x20-speedmode-carto1.wdf"
filename = folder_name + file_n

initialization = {'SliceValues': [None, None],  # Use None to count all
                  'NMF_NumberOfComponents': 6,
                  'PCA_components': 25,
                  # Put in the int number from 0 to _n_y:
                  'NumberOfLinesToSkip_Beggining': 0,
                  # Put in the int number from 0 to _n_y - previous element:
                  'NumberOfLinesToSkip_End': 0,
                  'BaselineCorrection': False,
                  'CosmicRayCorrection': True,
                  # Nearest neighbour method
                  # To use only in maps where step sizes are smaller then
                  # Sample's feature sizes (oversampled maps)
                  'AbsoluteScale': False,  # what type of colorbar to use
                  "save_data": False}
# %%
# Reading the data from the .wdf file
spectra, sigma, params, map_params, origins =\
    read_WDF(filename, verbose=True)
"""
-   `spectra` : 2D ndarray
        contains the intensities recorded at each point in a map scan.
        It is of shape: `(N°_measurement_points, N°_RamanShifts)`
-   `sigma` : 1D ndarray
        contains all the ramans shift values (x_values)
        Its' length is `N°_RamanShifts`
-   `params` is a dictionnary containing measurement parameters
-   `map_params` is dictionnary containing map parameters
-   `origins` : pandas dataframe
        gives detail on each point in the map
        (time of measurement, _coordinates and some other info).

> _Note: It should be noted that the timestamp
    recorded in the origins dataframe is in the Windows 64bit format,
    if you want to convert it to the human readable format,
    you can use the imported "convert_time" function_
"""
assert params['MeasurementType'] == 'Map', 'This script is intended for maps'


# %%
# put the retreived number of measurements in a variable
# with a shorter name, as it will be used quite often:
_n_points = params.get('Count', len(spectra))
try:
    # Finding in what axes the scan was taken:
    _scan_axes = np.where(map_params['NbSteps'] > 1)[0]
except (NameError, KeyError):
    _scan_axes = (0, 1)

try:
    # ATTENTION : from this point on in the script,
    # the two relevant dimensions will be called X and Y
    # regardless if one of them is Z in reality (for slices)
    _n_x, _n_y = map_params['NbSteps'][_scan_axes]
except (NameError, KeyError):
    while True:
        _n_x = int(input("Enter the total number of measurement points "
                         "along x-axis: "))
        _n_y = int(input("Enter the total number of measurement points "
                         "along y-axis: "))
        if _n_x*_n_y == _n_points:
            print("That looks ok.")
            break
        elif _n_x * _n_y != _n_points:
            warn("\nWrong number of points. Try again:")
            continue
        break

try:
    _s_x, _s_y = map_params['StepSizes'][_scan_axes]
except (NameError, KeyError):
    _s_x = int(input("Enter the size of the step along x-axis: "))
    _s_y = int(input("Enter the size of the step along y-axis: "))
    print("ok")


if (initialization['NumberOfLinesToSkip_Beggining']
        + initialization['NumberOfLinesToSkip_End']) > _n_y:
    raise SystemExit('You chose to skip more lines than present in the scan.\n'
                     'Please revise your initialization parameters')
# readjust the number of rows:
_n_y -= initialization['NumberOfLinesToSkip_End'] -\
    initialization['NumberOfLinesToSkip_Beggining']


spectra2 = np.copy(spectra)
spectra2.resize(_n_y, _n_x, len(sigma))
spectra = spectra2.reshape(_n_x*_n_y, -1)
spectra = pp.correct_saturated(spectra, (_n_y, _n_x))
del spectra2
# %%
# =============================================================================
#                               SLICING....
# =============================================================================
# Isolating the part of the spectra that interests us
spectra_kept, sigma_kept = ut.slice_lr(spectra, sigma, **initialization)
# Removing the lines from top and/or bottom of the map
spectra_kept = ut.skip_ud(spectra_kept, _n_x=_n_x, **initialization)

spectra_kept -= np.min(spectra_kept, axis=-1, keepdims=True)


spectra_log = (np.log(spectra_kept+1)).reshape(_n_y, _n_x, -1)
see_all_maps = ut.AllMaps(spectra_log,
                       sigma=sigma_kept)  # , title="raw spectra (log_scale)")
plt.suptitle("raw spectra (log_scale)")

# %%
# =============================================================================
# Finding the baseline using the asynchronous least squares method
# =============================================================================
if initialization['BaselineCorrection']:

    b_line = ut.baseline_als(spectra_kept, p=1e-3, lam=100*len(sigma_kept))

    b_corr_spectra = spectra_kept - b_line
    # Remove the eventual offsets:
    b_corr_spectra -= np.min(b_corr_spectra, axis=1)[:, np.newaxis]

    # Visualise the baseline correction:
    _baseline_stack = np.stack((spectra_kept, b_line, b_corr_spectra),
                               axis=-1)
    labels = ['original spectra', 'baseline', 'baseline corrected spectra']
    check_baseline = ut.NavigationButtons(sigma_kept, _baseline_stack,
                                       autoscale_y=True, label=labels)
    see_baseline_map = ut.AllMaps(b_line.reshape(_n_y, _n_x, -1),
                               sigma=sigma_kept)
    plt.suptitle("baseline")
else:
    b_corr_spectra = spectra_kept -\
        np.min(spectra_kept, axis=-1, keepdims=True)


# %%
# =============================================================================
#                                 CR correction...
# =============================================================================

if initialization['CosmicRayCorrection']:
    print("starting the cosmic ray correction")
    mock_sp3 = ut.remove_CRs(b_corr_spectra, sigma_kept,
                          _n_x=0, _n_y=0, **initialization)
else:
    mock_sp3 = b_corr_spectra


# %%
# =============================================================================
# ---------------------------------- PCA --------------------------------------
# =============================================================================
print(f"smoothing with PCA ({initialization['PCA_components']} components)")
# =============================================================================
mock_sp3 /= np.max(mock_sp3, axis=-1, keepdims=True)
pca = decomposition.PCA(n_components=initialization['PCA_components']+10)
spectra_reduced = pca.fit_transform(mock_sp3)
# spectra_reduced = np.dot(mock_sp3 - np.mean(mock_sp3, axis=0), pca.components_.T)

spectra_denoised = pca.inverse_transform(spectra_reduced)
spectra_denoised -= np.min(spectra_denoised, axis=-1, keepdims=True)
# spectra_denoised = np.dot(spectra_reduced, pca.components_)+np.mean(mock_sp3, axis=0)

vidji_pca = ut.AllMaps(spectra_reduced.reshape(_n_y, _n_x, -1),
                    components=pca.components_,
                    components_sigma=sigma_kept, title="pca component")



########### showing the smoothed spectra #####################
_s = np.stack((mock_sp3,
               spectra_denoised), axis=-1)
see_all_denoised = ut.NavigationButtons(sigma_kept, _s, autoscale_y=True,
                                     label=["scaled orig spectra",
                                            "pca denoised"],
                                     figsize=(12, 12))
see_all_denoised.figr.suptitle("PCA denoising result")

del b_corr_spectra
del mock_sp3
# %%
# =============================================================================
#                                   NMF step
# =============================================================================

_n_components = initialization['NMF_NumberOfComponents']
nmf_model = decomposition.NMF(n_components=_n_components, init='nndsvda',
                              max_iter=7, l1_ratio=0.5, alpha=2, regularization='components')
_start = time()
# print('starting nmf... (be patient, this may take some time...)')
mix = nmf_model.fit_transform(spectra_denoised)
components = nmf_model.components_
reconstructed_spectra = nmf_model.inverse_transform(mix)
_end = time()
print(f'nmf done in {_end - _start:.2f}s')

# %%
# =============================================================================
#                    preparing the mixture coefficients
# =============================================================================
_start_pos = 0

mix.resize((_n_x*_n_y), _n_components, )

mix = np.roll(mix, _start_pos, axis=0)

spectra_reconstructed = np.dot(mix, components)
_mix_reshaped = mix.reshape(_n_y, _n_x, _n_components)

vidju_nmf = ut.AllMaps(_mix_reshaped, components=components,
                    components_sigma=sigma_kept,
                    title="Map of component's contribution")
plt.suptitle("Resluts of NMF")
# %%
# =============================================================================
#                    Plotting the components....
# =============================================================================
sns.set()  # to make plots pretty :)

# to keep always the same colors for the same components:
col_norm = colors.Normalize(vmin=0, vmax=_n_components)
color_set = ScalarMappable(norm=col_norm, cmap="hsv")

# infer the number of subplots and their disposition from n_components
fi, _ax = plt.subplots(int(np.floor(np.sqrt(_n_components))),
                       int(np.ceil(_n_components /
                                   np.floor(np.sqrt(_n_components))
                                   )), sharex=True, sharey=False)
if _n_components > 1:
    _ax = _ax.ravel()
else:
    _ax = [_ax]
for _i in range(_n_components):
    _ax[_i].plot(sigma_kept, components[_i].T, color=color_set.to_rgba(_i))
    _ax[_i].set_title(f'Component {_i}')
    _ax[_i].set_yticks([])
try:
    fi.text(0.5, 0.04,
            f"{params['XlistDataType']} recordings"
            f" in {params['XlistDataUnits']} units",
            ha='center')
except:
    pass

# %%
# =============================================================================
#                       Plotting the main plot...
# =============================================================================
_n_fig_rows = int(np.floor(np.sqrt(_n_components)))
_n_fig_cols = int(np.ceil(_n_components / np.floor(np.sqrt(_n_components))))
fig, _ax = plt.subplots(_n_fig_rows, _n_fig_cols,
                        sharex=True, sharey=True)
if _n_components > 1:
    _ax = _ax.ravel()
else:
    _ax = [_ax]

mix_sum = np.sum(mix, axis=-1)


def onclick(event):
    '''Double-clicking on a pixel will pop-up the (cleaned) spectrum
    corresponding to that pixel, as well as its deconvolution on the components
    and again the reconstruction for visual comparison'''
    if event.inaxes:
        x_pos = int(np.floor(event.xdata))
        y_pos = int(np.floor(event.ydata))
        broj = int(y_pos*_n_x + x_pos)
        spec_num = int(y_pos*_n_x - _start_pos + x_pos)

        if event.dblclick:
            ff, aa = plt.subplots()
            aa.scatter(sigma_kept, spectra_denoised[spec_num], alpha=0.3,
                       label=f'(cleaned) spectrum n°{broj}')
            aa.plot(sigma_kept, spectra_reconstructed[broj], '--k',
                    label='reconstructed spectrum')
            for k in range(_n_components):
                aa.plot(sigma_kept, components[k]*mix[broj][k],
                        color=color_set.to_rgba(k),
                        label=f'Component {k} contribution'
                              f'({mix[broj][k]*100/mix_sum[broj]:.1f}%)')

# This next part is to reorganize the order of labels,
# so to put the scatter plot first
            handles, labels = aa.get_legend_handles_labels()
            order = list(np.arange(_n_components+2))
            new_order = [order[-1]]+order[:-1]
            aa.legend([handles[idx] for idx in new_order],
                      [labels[idx] for idx in new_order])
            aa.set_title(f'deconvolution of the spectrum from: '
                         f'line {y_pos} & column {x_pos}')
            ff.show()
    else:
        print("you clicked outside the canvas, you bastard :)")


_xcolumn_name, _ycolumn_name = ([origins.columns[i][1] for i in _scan_axes])

#################################################################################
############## This formatting should be adapted case by case ###################

#################################################################################
if initialization['AbsoluteScale'] == True:
    scaling = {'vmin': 0, 'vmax': 1}
else:
    scaling = {}
for _i in range(_n_components):
    sns.heatmap(_mix_reshaped[:, :, _i], ax=_ax[_i],
                cmap="Spectral_r", annot=False, **scaling)
#    _ax[_i].set_aspect(_s_y/_s_x)
    _ax[_i].set_title(f'Component {_i}', color=color_set.to_rgba(_i),
                      fontweight='extra bold')
    try:
        _x_ticks = origins.xs(_xcolumn_name, level=1,
                              axis=1).to_numpy().ravel()[:_n_x]
        _y_ticks = origins.xs(_ycolumn_name, level=1,
                              axis=1).to_numpy().ravel()[::_n_x]
        _pos_x = _ax[_i].get_xticks()
        _pos_y = _ax[_i].get_yticks()
        _xlabels = [str(x) for x in _x_ticks[::int(np.ceil(_n_x/len(_pos_x)))]]
        _ylabels = [str(y) for y in _y_ticks[::int(np.ceil(_n_y/len(_pos_y)))]]
        _ax[_i].set_xticklabels(_xlabels, rotation=45,
                                ha="right", fontstretch=50)
        _ax[_i].set_yticklabels(_ylabels, rotation=0,
                                ha="right", fontstretch=50)
        _ax[_i].set_xlabel(_xcolumn_name+"[µm]")
        _ax[_i].set_ylabel(_ycolumn_name+"[µm]")
    except:
        pass
fig.suptitle('Heatmaps showing the abundance of individual components'
             ' throughout the scanned area.')
fig.canvas.mpl_connect('button_press_event', onclick)

# %%
# =============================================================================
#        saving some data for usage in other software (Origin, Excel..)
# =============================================================================
if initialization["save_data"]:
    _basic_mix = pd.DataFrame(
        np.copy(mix),
        columns=[f"mixing coeff. for the component {l}"
                 for l in np.arange(mix.shape[1])]
    )
    _save_filename_extension = (f"_{_n_components}NMFcomponents_from"
                                f".csv")
    _save_filename_folder = '/'.join(x for x in filename.split('/')[:-1])+'/'\
                            + filename.split('/')[-1][:-4]+'/'
    if not os.path.exists(_save_filename_folder):
        os.mkdir(_save_filename_folder)

    _basic_mix.to_csv(
        f"{_save_filename_folder}MixingCoeffs{_save_filename_extension}",
        sep=';', index=False)
    _save_components = pd.DataFrame(
        components.T, index=sigma_kept,
        columns=[f"Component{_i}" for _i in np.arange(_n_components)])
    _save_components.index.name = 'Raman shift in cm-1'
    _save_components.to_csv(
        f"{_save_filename_folder}Components{_save_filename_extension}",
        sep=';')
