# -*- coding: utf-8 -*-
from wdfReader_new import convert_time, read_WDF
import deconvolution
import matplotlib.pyplot as plt
from sklearn import decomposition
from scipy import integrate
#import h5py
import numpy as np
import seaborn as sns; sns.set()
from timeit import default_timer as time
import pandas as pd
#from tkinter import filedialog, Tk, messagebox
'''This script uses Williams' script of deconvolution together with wdfReader to produce some informative graphical output.
ATTENTION: For the momoent, the scripts works only on  map scans (.wdf files)
All of the abovementioned scripts should be in your working directory (maybe you need to add the __init__.py file in the same folder as well.
You should first choose the data file with the map scan in the .wdf format (I could add later the input dialog)
You set the "snake" variable to "True" or "False" (depending on the way the map was recorded)
You choose the number of components. 
That's it
You should first get the plot of all the components found,
Then the heatmap of the mixing coefficients: when you double-click on a pixel on this map, 
it will pop-up another plot with the spectra recorded at this point, together with the contributions of each component
'''

'''Ideas to improve the script:
    Add the possiblity to plot the heatmap of the max_height of one selected peak (region) or it's area
    
    I should test Katia's idea, which is to do the deconvolultion on several subensembles of spectra,
    then on the whole spectra and then compare if the add up.
    
    Check out the issue of (non)normalized mixture coefficients.
    Would it be more meaningfull to renormalize the components (divide by max value, for exemple),
    so the mixture coefficients should add up to 1.
    (how would the division by max value reflect to mixture coeffs? - should they be then multiplied by the same value?)
    Now, each component has very different absolute values, 
    so even if the mixture coeff is small, the first component is nevertheless predominant
    in the decomosed spectra by the sheer fact that it's intensity is several orders of magnitude greater
    than that of some other components
    
    
'''
# -----------------------Choose a file--------------------------------------------
#filename = 'Data/Test-Na-SiO2 0079 -532nm-obj100-p100-10s over night carto.wdf'#scan_type 1, measurement_type 3
#filename = 'Data/Test-Na-SiO2 0079 droplet on quartz -532nm-obj50-p50-15s over night_Copy_Copy.wdf'#scan_type 2, measurement_type 2
#filename = 'Data/Test quartz substrate -532nm-obj100-p100-10s.wdf'#scan_type 2, measurement_type 1
#filename = 'Data/Hamza-Na-SiO2-532nm-obj100-p100-10s-extended-cartography - 1 accumulations.wdf'#scan_type 2 (wtf?), measurement_type 3
#filename = 'Data/M1SCMap_2_MJ_Truncated_CR2_NF50_PCA3_Clean2_.wdf'
#filename = 'Data/M1ANMap_Depth_2mm_.wdf'
#filename = 'Data/M1SCMap_depth_.wdf'
filename = 'Data/drop4.wdf'
#filename = 'Data/Sirine_siO21mu-plr-532nm-obj100-2s-p100-slice--10-10.wdf'


measure_params, map_params, sigma2, spectra, origins = read_WDF(filename)




#%% SLICING....

# one should always check if the spectra were recorded with the dead pixels included or not
# It turns out that firs 10 and the last 16 pixels on the Renishaw SVI spectrometer detector are reserved, 
# and no signal is recorded on those pixels by the detector. So we should either enter these parameters inside the Wire settings
# or if it's not done, remove those pixels here manually
# Furthermore, we sometimes want to perform the deconvolution only on a part of the spectra, so here you define the part that interests you
slice_values = (920,1120)# give your zone in cm-1

_coupe_bas = np.where(sigma2 == min(sigma2, key=lambda v: abs(slice_values[0]-v)))[0][0]
_coupe_haut = np.where(sigma2 == min(sigma2, key=lambda v: abs(slice_values[1]-v)))[0][0]
#x_axis_slice = slice(coupe_haut,coupe_bas)
x_axis_slice = slice(_coupe_haut,_coupe_bas) # you need to remember the order of the shifts is recorded from higher to lower
spectra_slice = np.index_exp[:,x_axis_slice] # Nothing to see here, move along

# Next few lines serve to isolate case-to-case file-specific problems in map scans:
if filename == 'Data/M1SCMap_2_MJ_Truncated_CR2_NF50_PCA3_Clean2_.wdf':
    slice_to_exclude = slice(5217,5499)
    slice_replacement = slice(4935,5217)
elif filename == 'Data/M1ANMap_Depth_2mm_.wdf': #Removing a few cosmic rays
    slice_to_exclude = np.index_exp[[10411, 10277, 17583]]
    slice_replacement = np.index_exp[[10412, 10278, 17584]]
elif filename == 'Data/drop4.wdf': #Removing a few cosmic rays manually
    slice_to_exclude = np.index_exp[[16021, 5554, 447, 14650, 16261, 12463, 14833, 13912, 5392, 11073, 16600, 20682, 2282, 18162, 20150, 12473, 4293, 16964, 19400]]
    slice_replacement = np.index_exp[[16020, 5555, 446, 14649, 16262, 12462, 14834, 13911, 5391, 11072,16601, 20683, 2283, 18163, 20151, 12474, 4294, 16965, 19401]]
    
elif filename == 'Data/Sirine_siO21mu-plr-532nm-obj100-2s-p100-slice--10-10.wdf': #Removing a few cosmic rays
    slice_to_exclude = np.index_exp[[1717, 11809, 2254, 3220, 6833]]
    slice_replacement = np.index_exp[[1718, 11808, 2255, 3221, 6832]]
else:
    slice_to_exclude = slice(None)
    slice_replacement = slice(None)



spektar3 = np.copy(spectra[spectra_slice])
sigma3 = np.copy(sigma2[x_axis_slice])
spektar3[slice_to_exclude] = np.copy(spektar3[slice_replacement])
spektar3 = spektar3[7007:]
if filename == 'Data/M1SCMap_2_MJ_Truncated_CR2_NF50_PCA3_Clean2_.wdf':
    
    zvrk = np.copy(spektar3.reshape(141,141,-1))
    zvrk[107:137,131:141,:] = zvrk[107:137,100:110,:] # patching the hole in the sample
    spektar3 = zvrk.reshape(141**2,-1)
#%% PCA...


pca = decomposition.PCA()
pca_fit = pca.fit(spektar3)
# =============================================================================
#   
# def choose_ncomp():
#     '''This plot serves as interface for selecting the number of components to use in NMF.
#     '''
#     variance_pourc = np.cumsum(pca_fit.explained_variance_ratio_)[:35]
#     test = 1e6*((-(variance_pourc**2) + 1))/np.sqrt((25-(np.arange(len(variance_pourc))-1)))
#     plt.scatter(np.arange(4,len(test)),test[4:]) # plots from 1 to 13 components only
#     plt.title("Double-click on the point to choose the number of components")#\n then middle-click to close the graph")
# #        plt.ylim(bottom=0.008928, top=0.00894)
#     plt.xlabel("number of principal components")
#     plt.ylabel("Variance % covered")
#     print(f'variance % covered with one single component is {variance_pourc[1]*100:.3f}%')
#     x = plt.ginput(2)
#     
#     for i in np.arange(len(x))[::-1]:
#         if(x[i]==x[i-1]): # double click
#             x_value = int(np.round(x[i][0]))
#             plt.close()
#             return x_value
# 
# =============================================================================
n_components = 2#choose_ncomp()
    
pca.n_components = n_components


denoised_spectra = pca.fit_transform(spektar3)

denoised_spectra = pca.inverse_transform(denoised_spectra)
print(f'The chosen number of components is: {n_components}')

cleaned_spectra = deconvolution.clean(sigma3, denoised_spectra, mode='area')

#%% NMF step


#cleaned_spectra = deconvolution.clean(sigma3, spektar3, mode='area')

start = time()
print('starting nmf... (be patient, this may take some time...)')
components, mix, nmf_reconstruction_error = deconvolution.nmf_step(cleaned_spectra, n_components, init='nndsvda')
end = time()
print(f'nmf done is {end-start:.3f}s')
#%%  MAP...

# Reading the values concerning the map:   
if measure_params['MeasurementType'] == 'Map':  
    x_index, y_index = np.where(map_params['NbSteps']>1)[0]
    n_x, n_y = map_params['NbSteps'][[x_index, y_index]]
    s_x, s_y = map_params['StepSizes'][map_params['StepSizes']>0]
    print('this is a map scan')
else:
    raise SystemExit('not a map scan')



#if map_params['MapAreaType'] == 'Slice':
#    y_points_nb = n_z
#else:
#    y_points_nb = n_y
    
mix.resize(n_x*n_y,n_components, )
comp_area = np.empty(n_components)
for z in range(n_components):
    comp_area[z] = integrate.trapz(components[z])# area beneath each component
    components[z] /= comp_area[z]# normalizing the components by area
    mix[:,z] *= comp_area[np.newaxis,z]# renormalizing the mixture coefficients
reconstructed_spectra = np.dot(mix, components)
novi_mix = mix.reshape(n_y,n_x,n_components)


#%% Plotting the components....
sns.set()
fi, ax = plt.subplots(int(np.floor(np.sqrt(n_components))), int(np.ceil(n_components/np.floor(np.sqrt(n_components)))))
if n_components > 1:
    ax = ax.ravel()
else:
    ax = [ax]
for i in range(n_components):
    ax[i].plot(sigma3, components[i].T)
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

#        line_px = n_y
        line_px=n_x # penser à changer pour slice
        broj = int(y_pos * line_px + x_pos)

        if event.dblclick:
            ff,aa = plt.subplots()
            aa.scatter(sigma3, cleaned_spectra[broj], alpha=0.3, label=f'(cleaned) spectrum n°{broj}')
            aa.plot(sigma3, reconstructed_spectra[broj], '--k', label='reconstructed spectrum')
            for k in range(n_components):
                aa.plot(sigma3, components[k]*mix[broj][k], label=f'Component {k} contribution ({mix[broj][k]*100:.1f}%)')
            
#this next part is to reorganize the order of labels, so to put the scatter plot first
            handles, labels = aa.get_legend_handles_labels()
            order = list(np.arange(n_components+2))
            new_order = [order[-1]]+order[:-1]
            aa.legend([handles[idx] for idx in new_order],[labels[idx] for idx in new_order])
            aa.set_title(f'spectra from {y_pos}th line and {x_pos}th column')
            ff.show()
    else:
        print("you clicked outside the canvas, you bastard :)")
y_ticks = [str(int(x)) for x in list(origins.iloc[:n_x*n_y:n_x,y_index+1])]
x_ticks = [str(int(x)) for x in list(origins.iloc[:n_x, x_index+1])]
for i in range(n_components):
    sns.heatmap(novi_mix[:,:,i], ax=ax[i], cmap="jet", annot=False)
#    ax[i].set_aspect(s_y/s_x)
    ax[i].set_title(f'Component {i}')
    plt.xticks(10*np.arange(np.floor(n_x/10)), x_ticks[::10], rotation=70)
    plt.yticks(10*np.arange(np.floor(n_y/10)), y_ticks[::10])
fig.text(0.5, 0.04, f"{origins.columns[x_index+1][1]} in {origins.columns[x_index+1][2]}", ha='center')
fig.text(0.04, 0.5, f"{origins.columns[y_index+1][1]} in {origins.columns[y_index+1][2]}")
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
