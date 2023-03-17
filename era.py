#!usr/evn/win/python 3.6 BY DZN 2018.06.27

"""
    This module focus on ECMWF wave mode wind data processing.
    Data downlink at:
    (http://apps.ecmwf.int/data-catalogues/era5/?stream=wave&expver=1&month=aug&param=other&year=2017&type=an&class=ea)
"""

# import
import os
import xarray as xr
import numpy as np
from math import radians
from datetime import datetime
# import matplotlib.pyplot as plt
# from mpl_toolkits.basemap import Basemap


#########################################################################
class ERA:
    def __init__(self, date=None, path=None):
        """
            date is expected date of ERA file want to read,
            flag is 'atmosphere' or 'ocean waves'.
        """
        self.pickfile(date, path)

    def pickfile(self, date, path):
        """ choose correct file, filename need to be confirm correct """
        file_list = os.listdir(path)
        if not file_list:
            return False
        file_list = [file for file in file_list if file.endswith('.nc')]
        for file in file_list:
            if date == datetime.strptime(file[:8], '%Y%m%d').date():
                self.filename = os.path.join(path, file)
                return True
        return False

    def readnc(self):
        """ Read ERA5 wind data with xarray """
        with xr.open_dataset(self.filename) as ds:
            # # compute wind speed and wind direction
            ds["WS"] = np.sqrt(ds.u10**2+ds.v10**2)
            wd = np.arctan2(ds.u10, ds.v10)/radians(1.0)
            wd.values[wd.values < 0.0] += 360.0
            ds["WD"] = wd
            self.data = ds.sortby("latitude")

    # def show(self, hour):
    #     lat = self.data.latitude
    #     lon = self.data.longitude
    #     wind = self.data.WS
    #     if hour < 0 or hour > 23:
    #         return
    #     plt.figure(figsize=(5.5, 2.2))
    #     m = Basemap(projection='cyl', llcrnrlat=-45, urcrnrlat=45,
    #                 llcrnrlon=-180, urcrnrlon=180, resolution='l')
    #     x, y = m(lon, lat)
    #     cs = plt.imshow(np.flipud(wind[hour]), cmap='YlOrRd',
    #                     extent=[x[0], x[-1], y[0], y[-1]])
    #     cbar = m.colorbar(cs, location="bottom", size='7%', pad='15%')
    #     cbar.set_clim(0, 30)
    #     cbar.set_ticks(np.linspace(0, 30, 6))
    #     cbar.set_label('Wind speed (m/s)')
    #     cbar.solids.set_edgecolor('face')
    #     cbar.draw_all()
    #     m.drawcoastlines()
    #     m.drawcountries()
    #     m.fillcontinents(color='darkgrey', lake_color='white')
    #     m.drawmapboundary(fill_color='white', linewidth=0.5)
    #     plt.title('ERA5 wind speed')
    #     plt.tight_layout()
    #     plt.show()

       
#########################################################################
if __name__ == '__main__':
    ear_path_atm = r'G:\ERA5\atmosphere'
    date = datetime(2017, 12, 1).date()
    era = ERA(date, ear_path_atm)
    era.readnc()
    # era.show(0)
