#!/usr/win/python 3.6 author:DZN ,2018
"""
====================================================================
Define class for record buoy data
====================================================================
"""

import os
import sys
import math
import numpy as np
import xarray as xr
import pickle
import datetime


# =============================================================================
def write_pickle(filename, data):
    """ write pickle file """
    try:
        pfile = open(filename, 'wb')
    except IOError:
        print('Open file Error!')
        sys.exit()
    pickle.dump(data, pfile)
    pfile.close()


# =============================================================================
def ncDataExplore(nc):
    """ print nc global attributions and variables """
    # # global attribution
    [print(finfo, ': ', getattr(nc, finfo)) for finfo in nc.ncattrs()]
    # # print variables keys
    [print(key) for key in nc.variables]
    # # variable objects  
    [print(key, ':\n', nc.variables[key], '\n') for key in nc.variables]
    nc.close()


# =============================================================================
def write_variables2file(nc, filename='temp.txt'):
    with open(filename, 'w') as pfile:
        [print(key, file=pfile) for key in nc.variables]
        [print(nc.variables[key], file=pfile) for key in nc.variables]


# =============================================================================
def list_files(path):
    """ Get file list for batch processing """
    if not os.path.isdir(path):    # the path is dir?
        print('Dir path Error!')
        sys.exit()
    # get nc file list of this path
    file_list = os.listdir(path)
    return file_list


# =============================================================================
def haversin(orgin, lat, lon):
    """
        calculate the distance between orgin and other points, which is
        recorded in lat, lon array.
    """
    EARTH_RADIUS = 6371.0   # unit:km
    nlat = np.radians(lat)  # change to radian
    delta_lat = abs(math.radians(orgin[0])-nlat)/2
    delta_lon = abs(math.radians(orgin[1])-np.radians(lon))/2
    temp = (np.sin(delta_lat)**2+math.cos(math.radians(orgin[0]))
            * np.cos(nlat)*np.sin(delta_lon)**2)
    return 2*EARTH_RADIUS*np.arcsin(np.sqrt(temp))


# =============================================================================
def shift_lon(lon):
    """ shift longitude range from [0,360] to [-180,180] """
    lon[lon > 180.0] -= 360.0  # basemap is -180 t0 180
    return lon


# =============================================================================
def convert_datetime(usert):
    return np.datetime64(usert, 'us').astype(datetime.datetime)


# =============================================================================
def data_combine(ds_dict):
    for num in ds_dict:
        try:
            data = xr.concat([data, ds_dict[num]],
                             dim="time",
                             coords="all")
        except UnboundLocalError:
            data = ds_dict[num]
    return data


# =============================================================================
def clear_dir(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            clear_dir(c_path)
        else:
            os.remove(c_path)

       
###############################################################################
def df_combine(data, x_var='wind_speed', y_var='WS',
                 flag='specific', type='validation'):
    """ data = {cygnss: DataFrame} """
    if 'specific' == flag:
        x = data[x_var]
        y = data[y_var]
        ind = y > 0.0
        x = x[ind]   # filter the NaN or miss data
        y = y[ind]
        return x, y
    elif 'all' == flag:
        total_data = data
        if type == 'validation':
            total_data = total_data[((total_data[x_var] >= 0.0)
                                    & (total_data[y_var] >= 0.0))]
        elif type == 'retrieval':
            total_data = total_data[total_data[y_var] >= 0.0]
        return total_data.dropna()