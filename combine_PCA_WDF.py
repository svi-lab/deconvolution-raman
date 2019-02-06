# -*- coding: utf-8 -*-
import wdfReader
import deconvolution
import matplotlib.pyplot as plt
from sklearn import decomposition
import h5py
import numpy as np
import seaborn as sns; sns.set()
from copy import copy
from timeit import default_timer as time
#from tkinter import filedialog, Tk, messagebox

#filename = 'Data/Test-Na-SiO2 0079 -532nm-obj100-p100-10s over night carto.wdf'#scan_type 1, measurement_type 3
#filename = 'Data/Test-Na-SiO2 0079 droplet on quartz -532nm-obj50-p50-15s over night_Copy_Copy.wdf'#scan_type 2, measurement_type 2
#filename = 'Data/Test quartz substrate -532nm-obj100-p100-10s.wdf'#scan_type 2, measurement_type 1
#filename = 'Data/Hamza-Na-SiO2-532nm-obj100-p100-10s-extended-cartography - 1 accumulations.wdf'#scan_type 2 (wtf?), measurement_type 3
#filename = 'Data/M1SCMap_2_MJ_Truncated_CR2_NF50_PCA3_Clean2_.wdf'
filename = 'Data/M1ANMap_Depth_2mm_.wdf'
#filename = 'Data/M1SCMap_depth_.wdf'
snake = True # The scanning mode: either always from left to right (snake = False), either left->right right->left (snake = True)



kiko = wdfReader.wdfReader(filename)

spektar = kiko.get_spectra()
sigma2 = kiko.get_xdata()
lambda_laser = 10000000/kiko.laser_wavenumber

# The script only works for map scans for the moment    
if kiko.measurement_type == 3:    
    map_type, mapa, n_x, n_y, n_z = kiko.get_map_area()
    print('this is a map scan')
else:
    raise SystemExit('not a map scan')

if snake == True:
    spektar1 = [spektar[((xx//n_x)+1)*n_x-(xx%n_x)-1] if (xx//n_x)%2==1 else spektar[xx] for xx in range(spektar.shape[0])]
    spektar2 = np.asarray(spektar1)
else:
    spektar2 = spektar
# Removing the outliers (as they were noticed in one of the scans, probably the firs file from those above)
spektar3 = copy(spektar2[:,575:950])#[:,10:-10])
sigma3 = copy(sigma2[575:950])#[10:-10])

if filename == 'Data/M1SCMap_2_MJ_Truncated_CR2_NF50_PCA3_Clean2_.wdf':
    # this is to remove the two lines with scanning problems:
    spektar3[5217:5499] = copy(spektar3[4935:5217])
#%%
start = time()

n_components=5
if False:#spektar3.shape[0] < spektar3.shape[1]:
    n_components, denoised_spectra = deconvolution.pca_step(spektar3)
else:
    pca = decomposition.PCA(n_components=n_components)
    #pca.fit(spektar3)
    #pca.n_components = n_components
    denoised_spectra = pca.fit_transform(spektar3)
    denoised_spectra = pca.inverse_transform(denoised_spectra)
end = time()
print(f'pca treatement done in {end-start:.3f}s')

start = time()

cleaned_spectra = deconvolution.clean(sigma3, denoised_spectra, mode='area')
end = time()
print(f'done cleaning in {end - start:.3f}s')
start = time()
print('starting nmf...')
components, mix, nmf_reconstruction_error = deconvolution.nmf_step(cleaned_spectra, n_components)
#reconstructed_spectra = np.dot(mix, components)
end = time()
print(f'nmf done is {end-start:.3f}s')
#%%
novi_mix = mix.reshape(n_y,n_x,n_components)
#%%
fi, ax = plt.subplots(int(np.ceil(np.sqrt(n_components))), int(np.ceil(np.sqrt(n_components/2))))
ax = ax.ravel()
for i in range(n_components):
    ax[i].plot(sigma3, components[i].T)
    ax[i].set_title(f'Component {i}')
#%%
#novi_mix[38] = copy(novi_mix[36])
fig, ax = plt.subplots(int(np.ceil(np.sqrt(n_components))), int(np.ceil(np.sqrt(n_components/2))))
ax = ax.ravel()
def onclick(event):
    if event.inaxes:
        a = event.inaxes
        for ii, axax in enumerate(ax):
            if axax == a:
                iii = ii
            
        x_pos = int(np.floor(event.xdata))
        y_pos = int(np.floor(event.ydata))

        line_px = n_y
        if False:#snake == True and y_pos%2 == 1: #this is to account for the impair lines, whose position should be counted backwards
            current_pos = line_px - x_pos
            print(f'impair line')
        else:
            current_pos = x_pos
        broj = int(y_pos * line_px + current_pos)
#        print(f'line_px={line_px}, current_pos={current_pos}')
#        print(f'you clicked on line {y_pos} on {iii}th canvas')
        if event.dblclick:
            ff,aa = plt.subplots()
            aa.plot(sigma3, cleaned_spectra[broj])
            for k in range(n_components):
                aa.plot(sigma3, components[k]*mix[broj][k])
            aa.set_title(f'spectra from {y_pos}th line and {x_pos}th column')
            aa.legend([f'(cleaned) spectre n°{broj}']+[f'Component {k} with mixing coeff of {mix[broj][k]:.3f}' for k in range(n_components)])
            ff.show()
    else:
        print("you clicked outside the canvas, you bastard :)")

for i in range(n_components):
    sns.heatmap(novi_mix[:,:,i], ax=ax[i], cmap="jet", annot=False)
    ax[i].set_title(f'Component {i}')
fig.canvas.mpl_connect('button_press_event', onclick)


#%%
# =============================================================================
# ag = deconvolution.area_graph_generator(sigma2, spektar, mix, components)
# 
# n_samples = kiko.count
# for i in range(0, n_samples, int(n_samples / 5)):
#     ag(i)
# 
# deconvolution.view_pca_denoising(sigma2, spektar, denoised_spectra)
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
    

