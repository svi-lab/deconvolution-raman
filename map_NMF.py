# -*- coding: utf-8 -*-
from read_WDF import convert_time, read_WDF
import deconvolution
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib import colors
from matplotlib.widgets import Button
from sklearn import decomposition
from scipy import integrate
#import h5py
import numpy as np
import seaborn as sns; sns.set()
from timeit import default_timer as time
import pandas as pd
#from tkinter import filedialog, Tk, messagebox
'''This script uses Williams' script of deconvolution read_WDF.py to produce some informative graphical output on map scans.
ATTENTION: For the momoent, the scripts works only on  map scans (from binary .wdf files)
All of the abovementioned scripts should be in your working directory (maybe you need to add the __init__.py file in the same folder as well.
You should first choose the data file with the map scan in the .wdf format (I could add later the input dialog)
You choose the number of components. 
That's it
You should first get the plot of all the components found,
Then the heatmap of the mixing coefficients: when you double-click on a pixel on this map, 
it will pop-up another plot with the spectra recorded at this point, together with the contributions of each component
'''

# -----------------------Choose a file--------------------------------------------
#filename = 'Data/Test-Na-SiO2 0079 -532nm-obj100-p100-10s over night carto.wdf'
#filename = 'Data/Test-Na-SiO2 0079 droplet on quartz -532nm-obj50-p50-15s over night_Copy_Copy.wdf'#scan_type 2, measurement_type 2
#filename = 'Data/Test quartz substrate -532nm-obj100-p100-10s.wdf'#scan_type 2, measurement_type 1
#filename = 'Data/Hamza-Na-SiO2-532nm-obj100-p100-10s-extended-cartography - 1 accumulations.wdf'#scan_type 2 (wtf?), measurement_type 3
filename = 'Data/M1SCMap_2_MJ_Truncated_CR2_NF50_PCA3_Clean2_.wdf'
#filename = 'Data/M1ANMap_Depth_2mm_.wdf'
#filename = 'Data/M1SCMap_depth_.wdf'
#filename = 'Data/drop4.wdf'
#filename = 'Data/Sirine_siO21mu-plr-532nm-obj100-2s-p100-slice--10-10.wdf'

initialization = {'SliceValues':[None,None], # Use None for taking the extremes
                  'NMF_NumberOfComponents':2, 
                  'NumberOfLinesToSkip_Beggining':0, # Put in the int number from 0 to _n_y
                  'NumberOfLinesToSkip_End':0} # Put in the int number from 0 to _n_y - previous element









measure_params, map_params, sigma, spectra, origins = read_WDF(filename) #reading the binary .wdf file
'''
"measure_params" is a dictionnary containing measurement parameters
"map_params" is dictionnary containing map parameters
"sigma" is a numpy array containing all the ramans shift values at which the intensities were recorded
"spectra" is a numpy array containing the intensities recorded at each point in a map scan. Its dimension is (number of points in map scan)x(len(sigma))
"origins" is a pandas dataframe giving detail on each point in the map scan (time of measurement, coordinates and some other info).
Remarque: It should be noted that the timestamp recorded in the origins dataframe is in the Windows 64bit format, 
if you want to convert it to the human readable format, you can use the imported "convert_time" function
'''
# Reading the values concerning the map:   
if measure_params['MeasurementType'] == 'Map':
    # Below we find indices of the axes in our map:
    _x_index, _y_index = np.where(map_params['NbSteps']>1)[0]
    # Thus you get the x-axis as the first one in the measurement map having more than 1 step recorded;
    # the other one is named y for the script purposes, even though it might be the depth in reality.
    _n_x, _n_y = map_params['NbSteps'][[_x_index, _y_index]]
    _s_x, _s_y = map_params['StepSizes'][[_x_index, _y_index]]
    print('this is a map scan')
else:
    raise SystemExit('not a map scan')
    
if initialization['NumberOfLinesToSkip_Beggining'] + initialization['NumberOfLinesToSkip_End'] > 0:
    raise SystemExit('You are skiping more lines than present in the scan.\nPlease revise your initialization parameters')
#%% Cosmic Rays:
'''This part is quite laborious at this stage, you should be much better off if you eliminate the cosmic rays beforehand using WiRE'''
_spectra1 = np.copy(spectra)
# Next few lines serve to isolate case-to-case file-specific problems in map scans:
if filename == 'Data/M1SCMap_2_MJ_Truncated_CR2_NF50_PCA3_Clean2_.wdf':
    _map_view = _spectra1.reshape(141,141,-1)
    _map_view[107:137,131:141,:] = _map_view[107:137,100:110,:] # patching the hole in the sample
    _spectra1 = _map_view.reshape(141**2,-1)
    _slice_to_exclude = slice(5217,5499)
    _slice_replacement = slice(4935,5217)
elif filename == 'Data/M1ANMap_Depth_2mm_.wdf': #Removing a few cosmic rays
    _slice_to_exclude = np.index_exp[[10411, 10277, 17583]]
    _slice_replacement = np.index_exp[[10412, 10278, 17584]]
elif filename == 'Data/drop4.wdf': #Removing a few cosmic rays manually
    _slice_to_exclude = np.index_exp[[16021, 5554, 447, 14650, 16261, 12463, 14833, 13912, 5392, 11073, 16600, 20682, 2282, 18162, 20150, 12473, 4293, 16964, 19400]]
    _slice_replacement = np.index_exp[[16020, 5555, 446, 14649, 16262, 12462, 14834, 13911, 5391, 11072,16601, 20683, 2283, 18163, 20151, 12474, 4294, 16965, 19401]]
#    _first_lines_to_skip = 79
#    _last_lines_to_skip = 20
elif filename == 'Data/Sirine_siO21mu-plr-532nm-obj100-2s-p100-slice--10-10.wdf': #Removing a few cosmic rays
    _slice_to_exclude = np.index_exp[[326, 700, 702, 1019, 1591, 1717, 2254, 3220, 3668, 3939, 5521, 6358, 6833, 6967, 7335, 7864, 8426, 10538, 10572, 11809]]
    _slice_replacement = np.index_exp[[327, 701, 703, 1020, 1592, 1718, 2255, 3221, 3669, 3940, 5522, 6359, 6832, 6968, 7336, 7865, 8427, 10539, 10573, 11808]]
else:
    _slice_to_exclude = slice(None)
    _slice_replacement = slice(None)



_spectra1[_slice_to_exclude] = np.copy(spectra[_slice_replacement])

#%% showing the raw spectra:
'''
This part allows us to scan trough spectra in order to visualize each spectrum individualy
'''
#plt.close('all')
figr, axr = plt.subplots()
plt.subplots_adjust(bottom=0.2)

s = np.copy(spectra)
n_points = int(measure_params['Capacity'])
s.resize(n_points, int(measure_params['PointsPerSpectrum']))
l, = plt.plot(sigma, s[0], lw=2)
plt.show()
class Index(object):
    ind = 0

    def next(self, event):
        self.ind += 1
        i = self.ind % n_points
        ydata = s[i]
        l.set_ydata(ydata)
        axr.relim()
        axr.autoscale_view(None, False, True)
        axr.set_title(f'spectrum number {i}')
        figr.canvas.draw()
        figr.canvas.flush_events()
        
    def next10(self, event):
        self.ind += 10
        i = self.ind % n_points
        ydata = s[i]
        l.set_ydata(ydata)
        axr.relim()
        axr.autoscale_view(None, False, True)
        axr.set_title(f'spectrum number {i}')
        figr.canvas.draw()
        figr.canvas.flush_events()
               
    def next100(self, event):
        self.ind += 100
        i = self.ind % n_points
        ydata = s[i]
        l.set_ydata(ydata)
        axr.relim()
        axr.autoscale_view(None, False, True)
        axr.set_title(f'spectrum number {i}')
        figr.canvas.draw()
        figr.canvas.flush_events()
        
    def next1000(self, event):
        self.ind += 1000
        i = self.ind % n_points
        ydata = s[i]		
        l.set_ydata(ydata)
        axr.relim()
        axr.autoscale_view(None, False, True)
        axr.set_title(f'spectrum number {i}')
        figr.canvas.draw()
        figr.canvas.flush_events()
        

    def prev(self, event):
        self.ind -= 1
        i = self.ind % n_points
        ydata = s[i]		
        l.set_ydata(ydata)
        axr.relim()
        axr.autoscale_view(None, False, True)
        axr.set_title(f'spectrum number {i}')
        figr.canvas.draw()
        figr.canvas.flush_events()
        
        
    def prev10(self, event):
        self.ind -= 10
        i = self.ind % n_points
        ydata = s[i]
        l.set_ydata(ydata)
        axr.relim()
        axr.autoscale_view(None, False, True)
        axr.set_title(f'spectrum number {i}')
        figr.canvas.draw()
        figr.canvas.flush_events()
        
        
    def prev100(self, event):
        self.ind -= 100
        i = self.ind % n_points
        ydata = s[i]
        l.set_ydata(ydata)
        axr.relim()
        axr.autoscale_view(None, False, True)
        axr.set_title(f'spectrum number {i}')
        figr.canvas.draw()
        figr.canvas.flush_events()
        
        
    def prev1000(self, event):
        self.ind -= 1000
        i = (self.ind) % n_points
        ydata = s[i]
        l.set_ydata(ydata)
        axr.relim()
        axr.autoscale_view(None, False, True)
        axr.set_title(f'spectrum number {i}')
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


#%% SLICING....
'''
One should always check if the spectra were recorded with the dead pixels included or not.
It is a parameter which should be set at the spectrometer configuration (Contact Renishaw for assistance)
As it turns out the first 10 and the last 16 pixels on the SVI Renishaw spectrometer detector are reserved, 
and no signal is ever recorded on those pixels by the detector.
So we should either enter these parameters inside the Wire settings
or, if it's not done, remove those pixels here manually

Furthermore, we sometimes want to perform the deconvolution only on a part of the spectra, so here you define the part that interests you
'''
_slice_values = initialization['SliceValues']# give your zone in cm-1

if not _slice_values[0]:
    _slice_values[0]=np.min(sigma)
if not _slice_values[1]:
    _slice_values[1]=np.max(sigma)
    
_condition = (sigma >= _slice_values[0]) & (sigma <= _slice_values[1])
sigma_kept = np.copy(sigma[_condition]) # adding np.copy if needed
spectra_kept = np.copy(_spectra1[:, _condition])




_first_lines_to_skip = initialization['NumberOfLinesToSkip_Beggining']
_last_lines_to_skip = initialization['NumberOfLinesToSkip_End']


_start_pos = _first_lines_to_skip*_n_x
if _last_lines_to_skip==0:
    _end_pos = None
else:
    _end_pos = -_last_lines_to_skip*_n_x

spectra_kept = spectra_kept[_start_pos:_end_pos]

coordinates = origins.iloc[_start_pos:_end_pos,[_x_index+1, _y_index+1]]   

#%% PCA...
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

    
pca.n_components = min(10, n_points, len(sigma_kept))#n_components


denoised_spectra = pca.fit_transform(spectra_kept)

denoised_spectra = pca.inverse_transform(denoised_spectra)


cleaned_spectra = deconvolution.clean(sigma_kept, denoised_spectra, mode='area')

#%% NMF step

n_components = initialization['NMF_NumberOfComponents']
print(f'The chosen number of components is: {n_components}')

start = time()
print('starting nmf... (be patient, this may take some time...)')
components, mix, nmf_reconstruction_error = deconvolution.nmf_step(cleaned_spectra, n_components, init='nndsvda')
basic_mix = pd.DataFrame(np.copy(mix), columns = [f"mixing coeff. for the component {l}" for l in np.arange(mix.shape[1])])
end = time()
print(f'nmf done is {end-start:.3f}s')
#%%  FOR THE HEATMAP...


  
mix.resize(_n_x*_n_y,n_components, )

mix = np.roll(mix, _start_pos, axis=0)
comp_area = np.empty(n_components)
for z in range(n_components):
    comp_area[z] = integrate.trapz(components[z])# area beneath each component
    components[z] /= comp_area[z]# normalizing the components by area
    mix[:,z] *= comp_area[np.newaxis,z]# renormalizing the mixture coefficients
reconstructed_spectra = np.dot(mix, components)
novi_mix = mix.reshape(_n_y,_n_x,n_components)


#%% Plotting the components....
sns.set()

col_norm = colors.Normalize(vmin=0, vmax=n_components)
color_set = ScalarMappable(norm=col_norm, cmap="brg")

fi, ax = plt.subplots(int(np.floor(np.sqrt(n_components))), int(np.ceil(n_components/np.floor(np.sqrt(n_components)))))
if n_components > 1:
    ax = ax.ravel()
else:
    ax = [ax]
for i in range(n_components):
    ax[i].plot(sigma_kept, components[i].T, color=color_set.to_rgba(i))
    ax[i].set_title(f'Component {i}')
    ax[i].set_yticks([])
fi.text(0.5, 0.04, f"{measure_params['XlistDataType']} recordings in {measure_params['XlistDataUnits']} units", ha='center')

#%% Plotting the main plot...
fig, ax = plt.subplots(int(np.floor(np.sqrt(n_components))), int(np.ceil(n_components/np.floor(np.sqrt(n_components)))), sharex=True, sharey=True)
if n_components > 1:
    ax = ax.ravel()
else:
    ax = [ax]
def onclick(event):
    '''Double-clicking on a pixel will pop-up the (cleaned) spectrum corresponding to that pixel, as well as it's deconvolution on the components
    and again the reconstruction for visual comparison'''
    if event.inaxes:
#        a = event.inaxes
#        for ii, axax in enumerate(ax):
#            if axax == a:
#                iii = ii
            
        x_pos = int(np.floor(event.xdata))
        y_pos = int(np.floor(event.ydata))

        broj = int(y_pos*_n_x + x_pos)
        spec_num = int(y_pos*_n_x - _start_pos + x_pos)

        if event.dblclick:
            ff,aa = plt.subplots()
            aa.scatter(sigma_kept, cleaned_spectra[spec_num], alpha=0.3, label=f'(cleaned) spectrum n°{broj}')
            aa.plot(sigma_kept, reconstructed_spectra[broj], '--k', label='reconstructed spectrum')
            for k in range(n_components):
                aa.plot(sigma_kept, components[k]*mix[broj][k], color=color_set.to_rgba(k), label=f'Component {k} contribution ({mix[broj][k]*100:.1f}%)')
            
#this next part is to reorganize the order of labels, so to put the scatter plot first
            handles, labels = aa.get_legend_handles_labels()
            order = list(np.arange(n_components+2))
            new_order = [order[-1]]+order[:-1]
            aa.legend([handles[idx] for idx in new_order],[labels[idx] for idx in new_order])
            aa.set_title(f'deconvolution of the spectrum from {y_pos}th line and {x_pos}th column')
            ff.show()
    else:
        print("you clicked outside the canvas, you bastard :)")
_xcolumn_name = ['X', 'Y', 'Z'][_x_index]
_ycolumn_name = ['X', 'Y', 'Z'][_y_index]
    
y_ticks = [str(int(x)) for x in np.asarray(origins[_ycolumn_name].iloc[:_n_x*_n_y:_n_x])]
x_ticks = [str(int(x)) for x in np.asarray(origins[_xcolumn_name].iloc[:_n_x])]
#y_ticks = [str(int(x)) for x in list(origins.iloc[:_n_x*_n_y:_n_x,_y_index+1])]
#x_ticks = [str(int(x)) for x in list(origins.iloc[:_n_x, _x_index+1])]
for i in range(n_components):
    sns.heatmap(novi_mix[:,:,i], ax=ax[i], cmap="jet", annot=False)
#    ax[i].set_aspect(_s_y/_s_x)
    ax[i].set_title(f'Component {i}', color=color_set.to_rgba(i), fontweight='extra bold')
    plt.xticks(10*np.arange(np.floor(_n_x/10)), x_ticks[::10])
    plt.yticks(10*np.arange(np.floor(_n_y/10)), y_ticks[::10])
fig.text(0.5, 0.014, f"{origins[_xcolumn_name].columns.to_frame().iloc[0,0]} in {origins[_xcolumn_name].columns.to_frame().iloc[0,1]}", ha='center')
fig.text(0.04, 0.5, f"{origins[_ycolumn_name].columns.to_frame().iloc[0,0]} in {origins[_ycolumn_name].columns.to_frame().iloc[0,1]}", rotation=90, va='center')
fig.suptitle('Heatmaps showing the abundance of individual components throughout the scanned area.')
fig.canvas.mpl_connect('button_press_event', onclick)


#%%

# =============================================================================
# subfolder = 'Data/Hamza/'
# components_df = pd.DataFrame(components, copy=True)
# spectra_df = pd.DataFrame(cleaned_spectra, copy=True)
# mix_df = pd.DataFrame(mix)
# 
# components_df.to_csv(subfolder+'Components.csv')
# spectra_df.to_csv(subfolder+'Spectra.csv')
# mix_df.to_csv(subfolder+'MixtureCoeffs.csv')
# =============================================================================
# =============================================================================
# with h5py.File('test.hdf5', 'w', libver='latest') as f:
# 
#     instrument_params = f.create_group("instrument_params")
#     measurement_params = f.create_group("measurement_params")
#     
#     map_points = f.create_group("map_points")
#     spectra = f.create_group("spectra")
#     
#     
#     spectra.create_dataset(name="recorded_intenisties", data=spektar)
# 
#     f.close()
# =============================================================================
save_filename_extension = f"_{n_components}components_RSfrom{_slice_values[0]:.1f}to{_slice_values[1]:.1f}_fromLine{_first_lines_to_skip}to{_n_y-_last_lines_to_skip if _last_lines_to_skip else 'End'}.csv"
save_coeff = pd.concat([coordinates, basic_mix], axis=1)
save_coeff.to_csv(f"{filename[:-4]}_MixingCoeffs{save_filename_extension}", index=False)
save_components = pd.DataFrame(components.T, index=sigma_kept, columns=[f"Component{i}" for i in np.arange(n_components)])
save_components.index.name = 'Raman shift in cm-1'
save_components.to_csv(f"{filename[:-4]}_Components{save_filename_extension}")
