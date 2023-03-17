#!/usr/win/python 3.6 by DZN, 2018

"""
====================================================================
This module is designed to match-filt the cygnss wind speed and the
buoy wind speed data provided by NBDC
====================================================================
"""

# import
import os
import pickle
import numpy as np


# =============================================================================
class Match_nwp(object):
    """ mutchup ECMWF ERA or GDAS wind and cygnss L2 wind """
    def __init__(self):
        """ 
        input data is single day data of CYGNSS and ECMWF
        wind_threshold is the comparison maximum wind,
        because ECMWF is for low to moderate wind speed
        """
        self.data = {}

    def matchup(self, data, nwp):
        """
            matchup the cygnss data and ECMWF ERA5 wave model
            wind analysis data, data = DataFrame
        """
        # # linear interploate the ECMWF wind
        lat = data.sp_lat
        lon = data.sp_lon
        lon[lon < 0.0] += 360.0
        ws = nwp.data.WS.interp(time=data.ddm_timestamp_utc,
                                latitude=lat,
                                longitude=lon,
                                kwargs={'fill_value': np.NaN})
        data["WS"] = ("time", ws.values)
        wd = nwp.data.WD.interp(time=data.ddm_timestamp_utc,
                                latitude=data.sp_lat,
                                longitude=data.sp_lon,
                                kwargs={'fill_value': np.NaN})
        data["WD"] = ("time", wd.values)
        data = data.dropna("time")
#           ind = ~np.isnat(data.time)
        data = data.isel(time=data.WS > 0.0)
        self.data = data.to_dataframe()

    def saveData(self, filename):
        with open(filename, 'wb') as pfile:
            pickle.dump(self.data, pfile)
