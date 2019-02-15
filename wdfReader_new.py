# -*- coding: latin-1 -*-
from __future__ import print_function
import struct
import numpy as np
import os
import time
import warnings

DATA_TYPES = ['Arbitrary','Spectral','Intensity','SpatialX','SpatialY','SpatialZ','SpatialR','SpatialTheta','SpatialPhi','Temperature','Pressure','Time','Derived','Polarization','FocusTrack','RampRate','Checksum','Flags','ElapsedTime','Frequency','MpWellSpatialX','MpWellSpatialY','MpLocationIndex','MpWellReference','PAFZActual','PAFZError','PAFSignalUsed','ExposureTime','EndMarker']
DATA_UNITS = ['Arbitrary','RamanShift','Wavenumber','Nanometre','ElectronVolt','Micron','Counts','Electrons','Millimetres','Metres','Kelvin','Pascal','Seconds','Milliseconds','Hours','Days','Pixels','Intensity','RelativeIntensity','Degrees','Radians','Celcius','Farenheit','KelvinPerMinute','FileTime','Microseconds','EndMarker']
SCAN_TYPES = ['Unspecified','Static','Continuous','StepRepeat','FilterScan','FilterImage','StreamLine','StreamLineHR','Point','MultitrackDiscrete','LineFocusMapping']
#filename = 'Data/Hamza-Na-SiO2-532nm-obj100-p100-10s-extended-cartography - 1 accumulations.wdf'
#filename = 'Data/M1SCMap_2_MJ_Truncated_CR2_NF50_PCA3_Clean2_.wdf'
#filename='Data/M1ANMap_Depth_2mm_.wdf'
#filename = 'Data/Sirine_siO21mu-plr-532nm-obj100-2s-p100-slice--10-101.wdf'
filename = 'Data/drop4.wdf'
#filename = 'Data/Sirine_siO21mu-plr-532nm-obj100-2s-p100-slice--10-10.wdf'
try:
    f = open(filename, 'rb')
except:
    raise ImportError('File not found. Check your filename.')

filesize = os.path.getsize(filename)

print(filename)
def _read(f=f, dtype=np.uint32, count=1):
    '''Reads bytes from binary file, with the most common values given as default.
    Returns the value itself if one value, or list if count > 1
    Note that you should do ".decode()" on strings to avoid getting strings like "b'string'"'''
    if count==1:
        return np.fromfile(f, dtype=dtype, count=count)[0]
    else:
        return np.fromfile(f, dtype=dtype, count=count)[0:count]

def convert_time(t):
    '''Takes the Windows 64bit timestamp and converts it to human readable format'''
    return time.strftime('%c', time.gmtime((t/1e7-11644473600)))

block_names=[]
block_sizes=[]
offset = 0
b_off = []
while offset<filesize-1:
    header_dt = np.dtype([('block_name','|S4'), ('block_id',np.int32), ('block_size',np.int64)])
    f.seek(offset)
    b_off.append(offset)
    block_header = np.fromfile(f, dtype=header_dt, count=1)
    offset += block_header['block_size'][0]
    block_names.append(block_header['block_name'][0].decode())
    block_sizes.append(block_header['block_size'][0])


name = 'WDF1'
params={}
gen = [i for i,x in enumerate(block_names) if x==name]
for i in gen:
    print(f"\n=============== Block : {name} ===============\nsize: {block_sizes[i]}, offset: {b_off[i]}")
    f.seek(b_off[i]+16)
    params['WdfFlag'] = _read(f,np.uint64)#['WdfXYXY','WdfChecksum','WdfCosmicRayRemoval','WdfMultitrack','WdfSaturation','WdfFileBackup','WdfTemporary','WdfSlice','WdfPQ'][_read(f,np.uint64)]
    f.seek(60)
    params['PointsPerSpectrum'] = npoints = _read(f)
    params['Capacity'] = nspectra = _read(f, np.uint64) # Number of spectra measured (nspectra)
    params['Count'] = ncollected =_read(f, np.uint64) # number of spectra written into the file (ncollected)
    params['AccumulationCount'] = _read(f) # number of accumulations per spectrum
    params['YlistLength'] = _read(f) # number of elements in the y-list (>1 for image)
    params['XlistLength'] = _read(f) # number of elements in the x-list
    params['DataOriginCount'] = _read(f) # number of data origin lists
    params['ApplicationName'] = _read(f, '|S24').decode()
    params['ApplicationVersion'] = _read(f, np.uint16, count=4)
    params['ScanType'] = SCAN_TYPES[_read(f)]
    params['MeasurementType'] = ['Unspecified', 'Single', 'Series', 'Map'][_read(f)]
    params['StartTime'] = convert_time(_read(f,np.uint64))
    params['EndTime'] = convert_time(_read(f,np.uint64))
    params['SpectralUnits'] = DATA_UNITS[_read(f)]
    params['LaserWaveLength'] = np.round(10e6/_read(f, '<f'),2)
    f.seek(240)
    params['Title'] = _read(f,'|S160').decode()
for key, val in params.items():
    print(f'{key} : \t{val}')
if nspectra != ncollected:
    warnings.warn(f'\nNot all spectra were recorded\nnspectra={nspectra}, while ncollected={ncollected}\nThe missing values will be filled with zeros.')

name = 'DATA'
gen = [i for i,x in enumerate(block_names) if x==name]
for i in gen:
    data_points_count=npoints*nspectra
    print(f"\n=============== Block : {name} ===============\nsize: {block_sizes[i]}, offset: {b_off[i]}")
    f.seek(b_off[i]+16)
    spectra = _read(f,'<f', count=data_points_count).reshape(nspectra, npoints)
    print(f'the shape of the spectra is: {spectra.shape}')
    

name = 'XLST'
gen = [i for i,x in enumerate(block_names) if x==name]
for i in gen:
    print(f"\n=============== Block : {name} ===============\nsize: {block_sizes[i]}, offset: {b_off[i]}")
    f.seek(b_off[i]+16)
    params['XlistDataType'] = DATA_TYPES[_read(f)]
    params['XlistDataUnits'] = DATA_UNITS[_read(f)]
    x_values = _read(f,'<f', count=npoints)
    
name = 'YLST' # This is where the image is stored (if recorded)
gen = [i for i,x in enumerate(block_names) if x==name]
for i in gen:
    print(f"\n=============== Block : {name} ===============\nsize: {block_sizes[i]}, offset: {b_off[i]}")
    f.seek(b_off[i]+16)
#    params['YlistDataType'] = DATA_TYPES[_read(f)]
#    params['YlistDataUnits'] = DATA_UNITS[_read(f)]
    y_values_count = int((block_sizes[i]-24)/4) # if > 1, we can say that this is the number of pixels in the recorded microscope image
    y_values = _read(f,'<f', count=y_values_count)
    
    
name='ORGN'
origin_labels = []
origin_set_dtypes = []
origin_set_units = []
origin_values = np.empty((params['DataOriginCount'],nspectra), dtype='<d')
gen = [i for i,x in enumerate(block_names) if x==name]
for i in gen:
    print(f"\n=============== Block : {name} ===============\nsize: {block_sizes[i]}, offset: {b_off[i]}")
    f.seek(b_off[i]+16)
    nb_origin_sets = _read(f)
    print(f"This is the params['DataOriginCount']={params['DataOriginCount']}, and this is the nb_origin_sets = {nb_origin_sets}")
    for set_n in range(nb_origin_sets):
        data_type_flag = _read(f).astype(np.uint16) # no idea why I had to add the astype part, but if I just read it as uint32, I got rubbish sometimes
        origin_set_dtypes.append(DATA_TYPES[data_type_flag])
        origin_set_units.append(DATA_UNITS[_read(f)])
        origin_labels.append(_read(f, '|S16').decode())
        if data_type_flag == 11:
            origin_values[set_n]=_read(f,np.uint64, count=nspectra)#.astype(np.int64)
        else:
            origin_values[set_n] = np.round(_read(f, '<d', count=nspectra),2)





name = 'WMAP'
map_params = {}
gen = [i for i,x in enumerate(block_names) if x==name]
for i in gen:
    print(f"\n=============== Block : {name} ===============\nsize: {block_sizes[i]}, offset: {b_off[i]}")
    f.seek(b_off[i]+16)
    m_flag = _read(f)
    map_params['MapAreaType'] = ['RandomPoints', 'ColumnMajor', 'Alternating', 'LineFocusMapping', 'InvertedRows', 'InvertedColumns', 'SurfaceProfile', 'XyLine', 'Slice'][m_flag+(8-m_flag)*(m_flag//128)]
    _read(f)
    map_params['InitialCoordinates'] = np.round(_read(f, '<f', count=3),2)
    map_params['StepSizes'] = np.round(_read(f, '<f', count=3),2)
    map_params['NbSteps'] = _read(f, np.uint32, count=3)
    map_params['LineFocusSize'] = _read(f)
for key, val in map_params.items():
    print(f'{key} : \t{val}')
    
print('\n\n\n')