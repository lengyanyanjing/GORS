import xarray as xr
import numpy as np
from utilities import shift_lon, convert_datetime


# =============================================================================
class GRData(object):
    """ extract DDMA, LES and other raw data from L1 files """
    def __init__(self, filename):
        """ Constructor """
        self.ocean_flag = 0.0
        self.rcg_threshold = 1.0
        self.variable_list = ["prn_code",
                              "track_id",
                              "sp_lat",
                              "sp_lon",
                              "sp_alt",
                              "sp_inc_angle",
                              "rx_to_sp_range",
                              "tx_to_sp_range",
                              "sp_rx_gain",
                              "ddm_nbrcs",
                              "ddm_les",
                            #   "nbrcs_scatter_area",
                            #   "les_scatter_area",
                            #   "brcs_ddm_sp_bin_delay_row",
                            #   "brcs_ddm_sp_bin_dopp_col",
                              "quality_flags"]
                            #   "power_analog",
                            #   "brcs"]
        self.open_data(filename)

    def open_data(self, filename):
        """ L1 file must be the netCDF4 format """
        with xr.open_dataset(filename) as ds:  # auto close file
            var_list = []
            for key in ds.data_vars:
                if key not in self.variable_list:
                    var_list.append(key)
            ds = ds.drop(var_list)
            self.clear_data(ds)

    def clear_data(self, ds):
        """ extract varibles """
        # # stack the dim of 'sample', 'ddm' to 'time'
        ds = ds.stack(time=("sample", "ddm"))
        # extract ocean data
        ds = ds.isel(time=ds.quality_flags == self.ocean_flag)
        ds = ds.dropna("time")
        ds["range_corr_gain"] = self.add_rcg(ds)
        # #drop element for RCG calculating
        ds = ds.drop(["sp_rx_gain", "rx_to_sp_range", "tx_to_sp_range"])
        ds = ds.isel(time=ds.range_corr_gain > self.rcg_threshold)
        if len(ds.time) == 0:
            self.data = None
            return
        ds.sp_lon.values = shift_lon(ds.sp_lon.values)
        date = convert_datetime(ds.ddm_timestamp_utc.values[0])
        ds.attrs["date"] = date.date()
        self.data = ds.sortby("time")
        self.add_scid()   # spacecraft in number

    def add_rcg(self, ds):
        """ aclculate the RCG of extract data """
        sp_rx_gain = 10**(ds.sp_rx_gain/10)
        inc_range = ds.rx_to_sp_range
        sca_range = ds.tx_to_sp_range
        return sp_rx_gain*1.e27/(inc_range*sca_range)**2

    def add_scid(self):
        """ space_craft id """
        for attr in self.data.attrs:
            if "platform" in attr:
                num_id = int(self.data.attrs[attr][26])
                time_dim = self.data.dims["time"]
                self.data.attrs['sc_num'] = num_id
                self.data['sc_num'] = ("time",
                                       num_id*np.ones(time_dim, dtype=int))
                break

