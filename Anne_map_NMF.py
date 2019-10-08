# -*- coding: utf-8 -*-
from read_WDF import convert_time, read_WDF
import deconvolution
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib import colors
from matplotlib.widgets import Button
from sklearn import decomposition
from scipy import integrate
import numpy as np
import seaborn as sns
from timeit import default_timer as time
import pandas as pd
import os
sns.set()
'''This script uses Williams' script of deconvolution read_WDF.py
to produce some informative graphical output on map/slice scans.
ATTENTION: For the moment, the scripts works only on map scans
(from binary .wdf files)
All of the abovementioned scripts should be in your working directory
You should first choose the data file with the map scan in the .wdf format
(I could add later the input dialog)
Set the initialization dictionary values
That'_s it!
First plot: your spectra (with navigation buttons)
Second plot: the components found
Third plot: the heatmap of the mixing coefficients
            (shows the abundance of each component troughout the map)
            when you double-click on a pixel on this map,
            it will pop-up another plot
            showing the spectra recorded at this point,
            together with the contributions of each component
'''

# -----------------------Choose a file-----------------------------------------
filename = 'Data/Anne/LE19266_lipp-static_x100_10s_P50_slicedepth1.wdf'
initialization = {'SliceValues': [100, 1300],  # Use None to count all
                  'NMF_NumberOfComponents': 4,
                  'PCA_components': 12,
                  # Put in the int number from 0 to _n_y:
                  'NumberOfLinesToSkip_Beggining': 0,
                  # Put in the int number from 0 to _n_y - previous element:
                  'NumberOfLinesToSkip_End': 0}

# Reading the data from the .wdf file
measure_params, map_params, sigma, spectra, origins = read_WDF(filename,
                                                               vebrose=True)
'''
"measure_params" is a dictionnary containing measurement parameters
"map_params" is dictionnary containing map parameters
"sigma" is a numpy array containing all the ramans shift values
     at which the intensities were recorded
"spectra" is a numpy array containing the intensities
    recorded at each point in a map scan.
    Its dimension is (number of points in map scan)x(len(sigma))
"origins" is a pandas dataframe giving detail on each point in the map scan
    (time of measurement, _coordinates and some other info).
Remarque: It should be noted that the timestamp
    recorded in the origins dataframe is in the Windows 64bit format,
    if you want to convert it to the human readable format,
    you can use the imported "convert_time" function
'''


# Reading the values concerning the map:
if measure_params['MeasurementType'] == 'Map':
    # Finding in what axes the scan was taken:
    _x_index, _y_index = np.where(map_params['NbSteps'] > 1)[0]
    # ATTENTION : from this point on in the script,
    # the two relevant dimensions  will be called
    # X and Y regardless if one of them is Z in reality (for slices)
    _n_x, _n_y = map_params['NbSteps'][[_x_index, _y_index]]
    _s_x, _s_y = map_params['StepSizes'][[_x_index, _y_index]]
else:
    raise SystemExit('not a map scan')

if (initialization['NumberOfLinesToSkip_Beggining']
        + initialization['NumberOfLinesToSkip_End']) > 0:
    raise SystemExit('You are skiping more lines than present in the scan.\n'
                     'Please revise your initialization parameters')

#%%
# =============================================================================
#          Manually removing the Cosmic Rays on a file to file basis:
# =============================================================================
'''This part is quite laborious at this stage,
you should be much better off if you eliminate the cosmic rays beforehand
using WiRE'''
_spectra1 = np.copy(spectra)

# The next few lines serve to isolate case-to-case file-specific problems
# in map scans:
if (filename == 'Data/Anne/LE19266_lipp-static_x100_10s_P50_slicedepth1.wdf'):
    _slice_to_exclude = np.index_exp[[63]]
    _slice_replacement = np.index_exp[[64]]
else:
    _slice_to_exclude = slice(None)
    _slice_replacement = slice(None)

_spectra1[_slice_to_exclude] = np.copy(spectra[_slice_replacement])

# %%
# =============================================================================
#                  showing the raw spectra:
# =============================================================================
'''
This part allows us to scan trough spectra in order to visualize each spectrum
individualy
'''
# plt.close('all')
figr, axr = plt.subplots()
plt.subplots_adjust(bottom=0.2)

_s = np.copy(spectra)
_n_points = int(measure_params['Capacity'])
_s.resize(_n_points, int(measure_params['PointsPerSpectrum']))
l, = plt.plot(sigma, _s[0], lw=2)
plt.show()


class Index(object):
    ind = 0

    def next(self, event):
        self.ind += 1
        _i = self.ind % _n_points
        ydata = _s[_i]
        l.set_ydata(ydata)
        axr.relim()
        axr.autoscale_view(None, False, True)
        axr.set_title(f'spectrum number {_i}')
        figr.canvas.draw()
        figr.canvas.flush_events()

    def next10(self, event):
        self.ind += 10
        _i = self.ind % _n_points
        ydata = _s[_i]
        l.set_ydata(ydata)
        axr.relim()
        axr.autoscale_view(None, False, True)
        axr.set_title(f'spectrum number {_i}')
        figr.canvas.draw()
        figr.canvas.flush_events()

    def next100(self, event):
        self.ind += 100
        _i = self.ind % _n_points
        ydata = _s[_i]
        l.set_ydata(ydata)
        axr.relim()
        axr.autoscale_view(None, False, True)
        axr.set_title(f'spectrum number {_i}')
        figr.canvas.draw()
        figr.canvas.flush_events()

    def next1000(self, event):
        self.ind += 1000
        _i = self.ind % _n_points
        ydata = _s[_i]
        l.set_ydata(ydata)
        axr.relim()
        axr.autoscale_view(None, False, True)
        axr.set_title(f'spectrum number {_i}')
        figr.canvas.draw()
        figr.canvas.flush_events()

    def prev(self, event):
        self.ind -= 1
        _i = self.ind % _n_points
        ydata = _s[_i]
        l.set_ydata(ydata)
        axr.relim()
        axr.autoscale_view(None, False, True)
        axr.set_title(f'spectrum number {_i}')
        figr.canvas.draw()
        figr.canvas.flush_events()

    def prev10(self, event):
        self.ind -= 10
        _i = self.ind % _n_points
        ydata = _s[_i]
        l.set_ydata(ydata)
        axr.relim()
        axr.autoscale_view(None, False, True)
        axr.set_title(f'spectrum number {_i}')
        figr.canvas.draw()
        figr.canvas.flush_events()

    def prev100(self, event):
        self.ind -= 100
        _i = self.ind % _n_points
        ydata = _s[_i]
        l.set_ydata(ydata)
        axr.relim()
        axr.autoscale_view(None, False, True)
        axr.set_title(f'spectrum number {_i}')
        figr.canvas.draw()
        figr.canvas.flush_events()

    def prev1000(self, event):
        self.ind -= 1000
        _i = (self.ind) % _n_points
        ydata = _s[_i]
        l.set_ydata(ydata)
        axr.relim()
        axr.autoscale_view(None, False, True)
        axr.set_title(f'spectrum number {_i}')
        figr.canvas.draw()
        figr.canvas.flush_events()


callback = Index()

axprev1000 = plt.axes([0.097, 0.05, 0.1, 0.04])
axprev100 = plt.axes([0.198, 0.05, 0.1, 0.04])
axprev10 = plt.axes([0.299, 0.05, 0.1, 0.04])
axprev1 = plt.axes([0.4, 0.05, 0.1, 0.04])
axnext1 = plt.axes([0.501, 0.05, 0.1, 0.04])
axnext10 = plt.axes([0.602, 0.05, 0.1, 0.04])
axnext100 = plt.axes([0.703, 0.05, 0.1, 0.04])
axnext1000 = plt.axes([0.804, 0.05, 0.1, 0.04])

bprev1000 = Button(axprev1000, 'Prev.1000')
bprev1000.on_clicked(callback.prev1000)
bprev100 = Button(axprev100, 'Prev.100')
bprev100.on_clicked(callback.prev100)
bprev10 = Button(axprev10, 'Prev.10')
bprev10.on_clicked(callback.prev10)
bprev = Button(axprev1, 'Prev.1')
bprev.on_clicked(callback.prev)
bnext = Button(axnext1, 'Next1')
bnext.on_clicked(callback.next)
bnext10 = Button(axnext10, 'Next10')
bnext10.on_clicked(callback.next10)
bnext100 = Button(axnext100, 'Next100')
bnext100.on_clicked(callback.next100)
bnext1000 = Button(axnext1000, 'Next1000')
bnext1000.on_clicked(callback.next1000)

# %%
# =============================================================================
#                               SLICING....
# =============================================================================
'''
One should always check if the spectra were recorded with the dead pixels
included or not. It is a parameter which should be set at the spectrometer
configuration (Contact Renishaw for assistance)
As it turns out the first 10 and the last 16 pixels on the SVI Renishaw
spectrometer detector are reserved, and no signal is ever recorded on those
pixels by the detector. So we should either enter these parameters inside the
Wire settings or, if it'_s not done, remove those pixels here manually

Furthermore, we sometimes want to perform the deconvolution only on a part
of the spectra, so here isolate only the part that interests you
'''
_slice_values = initialization['SliceValues']  # give your zone in cm-1

if not _slice_values[0]:
    _slice_values[0] = np.min(sigma)
if not _slice_values[1]:
    _slice_values[1] = np.max(sigma)

_condition = (sigma >= _slice_values[0]) & (sigma <= _slice_values[1])
sigma_kept = np.copy(sigma[_condition])  # adding np.copy if needed
spectra_kept = np.copy(_spectra1[:, _condition])

_first_lines_to_skip = initialization['NumberOfLinesToSkip_Beggining']
_last_lines_to_skip = initialization['NumberOfLinesToSkip_End']

_start_pos = _first_lines_to_skip * _n_x
if _last_lines_to_skip == 0:
    _end_pos = None
else:
    _end_pos = -_last_lines_to_skip * _n_x

spectra_kept = spectra_kept[_start_pos:_end_pos]

_coordinates = origins.iloc[_start_pos:_end_pos, [_x_index+1, _y_index+1]]

# %%
from scipy import sparse
from scipy.sparse.linalg import spsolve, inv

def baseline_als2(y, lam=10e4, p=10e-5, niter=12):
  L = len(y)
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  for i in np.arange(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

try:
    spectra_kept
except NameError:
    spectra_kept = np.copy(spectra)
try:
    sigma_kept
except NameError:
    sigma_kept = np.copy(sigma)





#%%
# =============================================================================
#                                     PCA...
# =============================================================================
try:
    spectra_kept
except NameError:
    spectra_kept = np.copy(spectra)
try:
    sigma_kept
except NameError:
    sigma_kept = np.copy(sigma)

pca = decomposition.PCA()
pca_fit = pca.fit(spectra_kept)
pca.n_components = min(initialization['PCA_components'],
                       _n_points,
                       len(sigma_kept))
spectra_denoised = pca.fit_transform(spectra_kept)
spectra_denoised = pca.inverse_transform(spectra_denoised)
spectra_cleaned = deconvolution.clean(sigma_kept, spectra_denoised,
                                      mode='area')

# %%
# =============================================================================
#                                   NMF step
# =============================================================================

_n_components = initialization['NMF_NumberOfComponents']
# =============================================================================
# _start = time()
# print('starting nmf... (be patient, this may take some time...)')
# components, mix, nmf_reconstruction_error = \
#     deconvolution.nmf_step(spectra_cleaned, _n_components,
#                            init='nndsvdar', beta=30000000)
#
#
# _end = time()
# print(f'nmf done is {_end-_start:.3f}_s')
# =============================================================================
_n_components = 4
nmf  = decomposition.NMF(_n_components,
                         init='nndsvda', alpha=0.006)
mix = nmf.fit_transform(spectra_cleaned)
components = nmf.components_
nmf_reconstruction_error = nmf.reconstruction_err_
# %%
# =============================================================================
#                    preparing the mixture coefficients
# =============================================================================

mix.resize(_n_x*_n_y, _n_components, )

mix = np.roll(mix, _start_pos, axis=0)
_comp_area = np.empty(_n_components)
for _z in range(_n_components):
    # area beneath each component:
    _comp_area[_z] = integrate.trapz(components[_z])
    components[_z] /= _comp_area[_z]  # normalizing the components by area
    # renormalizing the mixture coefficients:
    mix[:, _z] *= _comp_area[np.newaxis, _z]
spectra_reconstructed = np.dot(mix, components)
_mix_reshaped = mix.reshape(_n_y, _n_x, _n_components)


# %%
# =============================================================================
#                    Plotting the components....
# =============================================================================
sns.set()  # to make plots pretty :)

col_norm = colors.Normalize(vmin=0, vmax=_n_components)
color_set = ScalarMappable(norm=col_norm, cmap="brg")

fi, _ax = plt.subplots(int(np.floor(np.sqrt(_n_components))),
                       int(np.ceil(_n_components /
                                   np.floor(np.sqrt(_n_components))
                                   )))
if _n_components > 1:
    _ax = _ax.ravel()
else:
    _ax = [_ax]
for _i in range(_n_components):
    _ax[_i].plot(sigma_kept, components[_i].T, color=color_set.to_rgba(_i))
    _ax[_i].set_title(f'Component {_i}')
    _ax[_i].set_yticks([])
fi.text(0.5, 0.04,
        f"{measure_params['XlistDataType']} recordings"
        f"in {measure_params['XlistDataUnits']} units",
        ha='center')

# %%
# =============================================================================
#                       Plotting the main plot...
# =============================================================================
fig, _ax = plt.subplots(int(np.floor(np.sqrt(_n_components))),
                        int(np.ceil(_n_components /
                                    np.floor(np.sqrt(_n_components))
                                    )),
                        sharex=True, sharey=True)
if _n_components > 1:
    _ax = _ax.ravel()
else:
    _ax = [_ax]


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
            aa.scatter(sigma_kept, spectra_cleaned[spec_num], alpha=0.3,
                       label=f'(cleaned) spectrum n°{broj}')
            aa.plot(sigma_kept, spectra_reconstructed[broj], '--k',
                    label='reconstructed spectrum')
            for k in range(_n_components):
                aa.plot(sigma_kept, components[k]*mix[broj][k],
                        color=color_set.to_rgba(k),
                        label=f'Component {k} contribution'
                              f'({mix[broj][k]*100:.1f}%)')

# This next part is to reorganize the order of labels,
# so to put the scatter plot first
            handles, labels = aa.get_legend_handles_labels()
            order = list(np.arange(_n_components+2))
            new_order = [order[-1]]+order[:-1]
            aa.legend([handles[idx] for idx in new_order],
                      [labels[idx] for idx in new_order])
            aa.set_title(f'deconvolution of the spectrum from'
                         f'{y_pos}th line and {x_pos}th column')
            ff.show()
    else:
        print("you clicked outside the canvas, you bastard :)")


_xcolumn_name = ['X', 'Y', 'Z'][_x_index]
_ycolumn_name = ['X', 'Y', 'Z'][_y_index]

tsx = 2  # to scale the denisty of the ticks
tsy = 1
_y_ticks = [str(int(x)) for x in
            np.asarray(origins[_ycolumn_name].iloc[:_n_x*_n_y:_n_x])]
_x_ticks = [str(int(x)) for x in
            np.asarray(origins[_xcolumn_name].iloc[:_n_x])]

for _i in range(_n_components):
    sns.heatmap(_mix_reshaped[:, :, _i], ax=_ax[_i], cmap="jet", annot=False)
#    _ax[_i].set_aspect(_s_y/_s_x)
    _ax[_i].set_title(f'Component {_i}', color=color_set.to_rgba(_i),
                      fontweight='extra bold')
    plt.xticks(tsx*np.arange(np.floor(_n_x/tsx)), _x_ticks[::tsx])
    plt.yticks(tsy*np.arange(np.floor(_n_y/tsy)), _y_ticks[::tsy])
fig.text(0.5, 0.014,
         f"{origins[_xcolumn_name].columns.to_frame().iloc[0,0]} in"
         f"{origins[_xcolumn_name].columns.to_frame().iloc[0,1]}",
         ha='center')
fig.text(0.04, 0.5,
         f"{origins[_ycolumn_name].columns.to_frame().iloc[0,0]} in"
         f"{origins[_ycolumn_name].columns.to_frame().iloc[0,1]}",
         rotation=90, va='center')
fig.suptitle('Heatmaps showing the abundance of individual components'
             'throughout the scanned area.')
fig.canvas.mpl_connect('button_press_event', onclick)


# %%
# =============================================================================
#        saving some data for usage in other software (Origin, Excel..)
# =============================================================================
_basic_mix = pd.DataFrame(
        np.copy(mix),
        columns=[f"mixing coeff. for the component {l}"
                 for l in np.arange(mix.shape[1])]
        )
_save_filename_extension = (f"_{_n_components}components_RSfrom"
                            f"{_slice_values[0]:.1f}to{_slice_values[1]:.1f}"
                            f"_fromLine{_first_lines_to_skip}to"
                            f"{_n_y - _last_lines_to_skip if _last_lines_to_skip else 'End'}"
                            f".csv")
_save_filename_folder = '/'.join(x for x in filename.split('/')[:-1])+'/'\
                        + filename.split('/')[-1][:-4]+'/'
if not os.path.exists(_save_filename_folder):
    os.mkdir(_save_filename_folder)

_save_coeff = pd.concat([_coordinates, _basic_mix], axis=1)
_save_coeff.to_csv(
        f"{_save_filename_folder}MixingCoeffs{_save_filename_extension}",
        sep=';', index=False)
_save_components = pd.DataFrame(
        components.T, index=sigma_kept,
        columns=[f"Component{_i}" for _i in np.arange(_n_components)])
_save_components.index.name = 'Raman shift in cm-1'
_save_components.to_csv(
        f"{_save_filename_folder}Components{_save_filename_extension}",
        sep=';')
