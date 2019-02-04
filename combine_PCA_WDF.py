# -*- coding: utf-8 -*-
import wdfReader
import matplotlib.pyplot as plt
import h5py
import numpy as np
import seaborn as sns; sns.set()
#from tkinter import filedialog, Tk, messagebox

filename = 'Test-Na-SiO2 0079 -532nm-obj100-p100-10s over night carto.wdf'

kiko = wdfReader.wdfReader(filename)

spektar = kiko.get_spectra()

sigma2 = kiko.get_xdata()

lambda_laser = 10000000/kiko.laser_wavenumber


# =============================================================================
# plt.close('all')
# plt.figure()
# 
# for i in  range(kiko.count):
# 
#     plt.plot(sigma, spectra[i])
# =============================================================================
    

map_type, mapa, n_x, n_y, n_z = kiko.get_map_area()

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
    
#%%
x_grid, y_grid = np.mgrid[(mapa[0]):(n_x*mapa[3]):n_x*1j, (mapa[1]):(n_x*mapa[4]):n_y*1j]

x_grid2, y_grid2 = np.ogrid[(mapa[0]):(n_x*mapa[3]):n_x*1j, (mapa[1]):(n_x*mapa[4]):n_y*1j]
#%%
reshaped_spectra = spektar.reshape(n_y,n_x,-1)

moyenne = np.mean(reshaped_spectra, axis=2)

xticks = np.round(x_grid2.squeeze(), decimals=2)
yticks = np.round(y_grid2.squeeze(), decimals=2)
sns.heatmap(moyenne, xticklabels=xticks, yticklabels=yticks)




