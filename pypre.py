import os
import sys
import pickle
import numpy as np
from collections import defaultdict
from datetime import datetime
from datetime import date as datetype
import pandas as pd
from pygnssr import GRData
from utilities import write_pickle, data_combine
from era import ERA
from match import Match_nwp
from model import data_specular_filter, compute_effective_area

# import warnings
# warnings.filterwarnings(action='ignore', module='numpy')

"""
    This module is for wind retrieval prparing, including check the
    dataset files, extract neccessary data saving in temperary files. 
"""


np.warnings.filterwarnings('ignore')    # ingnor the invalide value warning


###############################################################################
def load_data(obs):
    """ load parametric gmf files """
    if isinstance(obs, str):
        with open(obs, 'rb') as pfile:
            return pickle.load(pfile)
    # direct input data
    elif isinstance(obs, pd.DataFrame):
        return obs


#########################################################################
def dataset_info(path):
    """
        return the information of L1 or L2 files, including the range of data,
        number of each day
    """
    if not os.path.exists(path):
        print('Dataset directoy does not exist!')
        return
    dir_list = os.listdir(path)   # file list of each day L1 or L2 file
    dir_list.sort()
    start_date = datetime.strptime(dir_list[0], '%Y%j')  # start date of all dataset
    end_date = datetime.strptime(dir_list[-1], '%Y%j')   # end date of all dataset
    print('Dataset start from: {}, DOY: {}'.format(start_date.strftime('%Y/%m/%d'), dir_list[0]))
    print('Dataset end at: {}, DOY: {}'.format(end_date.strftime('%Y/%m/%d'), dir_list[-1]))
    df = pd.DataFrame(columns=['Calender', 'DOY', 'Num. of files'])
    for i, doy in enumerate(dir_list):
        # # create datetime object with DOY
        date = datetime.strptime(doy, '%Y%j')
        file_path = os.path.join(path, doy)
        # # list L1 files
        file_list = os.listdir(file_path)
        # # add to DataFrame object
        df.loc[i] = {'Calender': date.strftime('%Y/%m/%d'),
                     'DOY': date.timetuple().tm_yday,
                     'Num. of files': len(file_list) if file_list else 0}
    return df


#########################################################################
def load_cyg_dataset(path, flag='date', start=None, end=None):
    """
        collect the cygnss L1 or L2 data, recording in dict, return dict.
        paras:
        --------------------
        path: L1 or L2 files saving directory.
        flag: set data extracting range with 'doy' of 'date'
        start: dataset start DOY of date
        end: dataset end DOY or date

        Notes: start or end date do not out of range of dataset
    """
    if not os.path.exists(path):
        print('Dataset directoy does not exist!')
        sys.exit()
    file_manager = defaultdict(list)
    dir_list = os.listdir(path)  # list folders on current directory
    dir_list.sort()
    # date select
    if flag == 'doy':
        if (len(start) != 7) or (len(end) != 7):
            print('DOY format is incorrect, please check!')
            return
        if (start in dir_list) and (end in dir_list):
            dir_list = dir_list[dir_list.index(start):dir_list.index(end)+1]
        else:
            print('Dataset index out of range!')
    elif flag == 'date':
        if isinstance(start, str) and isinstance(end, str):
            start_date = datetime.strptime(start, '%Y/%m/%d')
            start_doy = start_date.strftime('%Y')+start_date.strftime('%j')
            end_date = datetime.strptime(end, '%Y/%m/%d')
            end_doy = end_date.strftime('%Y')+end_date.strftime('%j')
            if (start_doy in dir_list) and (end_doy in dir_list):
                dir_list = dir_list[dir_list.index(start_doy):dir_list.index(end_doy)+1]
            else:
                print('Dataset index out of range!')
        if isinstance(start, datetime) and isinstance(end, datetime):
            start_doy = start.strftime('%Y')+start.strftime('%j')
            end_doy = end.strftime('%Y')+end.strftime('%j')
            if (start_doy in dir_list) and (end_doy in dir_list):
                dir_list = dir_list[dir_list.index(start_doy):dir_list.index(end_doy)+1]
            else:
                print('Dataset index out of range!')
    for doy in dir_list:
        file_path = os.path.join(path, doy)
        # list L1 files
        file_list = os.listdir(file_path)
        [file_manager[file_path].append(file) for file in file_list
         if os.path.isfile(os.path.join(file_path, file))]
    return file_manager


#########################################################################
def output_file_info(file_manager, path=os.getcwd(), filename='filelog.txt'):
    """ write the cyg l1 files information to the file  """
    if not os.path.exists(path):
        os.mkdir(path)
    filename = os.path.join(path, filename)
    with open(filename, 'w') as pfile:
        for path in file_manager:
            print(path, '\n', file=pfile)
            [print(filename, file=pfile) for filename in file_manager[path]]
        print('\n', file=pfile)


#########################################################################
def compute_average(count, num, time_series):
    """
        calculate the index for time average.
        paras:
        ------------------
        count: location index of DataFrame
        num: number of time average epoches
        time_array: DataFrame.time_stamp
        
        return:
        ------------------
        index in list or scale
    """
    if num <= 3:
            return low_equal_count(num, count, time_series)
    else:
        return over_count(num, count, time_series)


def low_equal_count(num, count, time_series):
    """ calculate time average, the number of epoch low less or equal 3 """
    if ((count == 0) or (2 == num) or (time_difference(time_series.iloc[count],
                                       time_series.iloc[count-1]) > 1.0)):
        return range(count, count+2)
    else:
        return range(count-1, count+2)


def over_count(num, count, time_series):
    """ calculate time average, the number of epoch over 3 """
    try:  # count +2 may occure out of boundary
        if (time_difference(time_series.iloc[count+2],
            time_series.iloc[count+1]) > 1.0):  # occure an epoch interval
            return low_equal_count(3, count, time_series)
        else:
            # the first epoch or occure  interval after the first sample
            if ((0 == count) or (time_difference(time_series.iloc[count],
                time_series.iloc[count-1]) > 1.0)):
                return range(count, count+2)
            # #occure an epoch interval after the second sample
            # #or the number of average epoch is 4
            elif ((1 == count) or (num == 4) or
                  (time_difference(time_series.iloc[count-1],
                   time_series.iloc[count-2]) > 1.0)):
                return range(count-1, count+3)
            else:
                # average 5 epoches
                return range(count-2, count+3)
    except IndexError:
        # count will not point to the before-terminal
        return low_equal_count(3, count, time_series)


def time_difference(ltime, rtime):
    """ calculate time difference in second """
    return (ltime-rtime).total_seconds()


def get_num_average_epoch(incidence):
    """
        calculate the time average number of specific incidence of specular
        paras:
        ---------------
        incidence: scale
        return:
        ---------------
        time average number
    """
    number = (5, 4, 3, 2, 1)           # time average number
    nodes = (17.0, 31.0, 41.0, 48.0)   # piecewise incidence
    for i, node in enumerate(nodes):
        if incidence <= node:
            return number[i]
    # incidence over than 48 degree
    if incidence > nodes[-1]:
        return number[-1]


def time_averaging(data):
    """
    time averaging to the collected data
        paras:
        ---------------
        data_dict: {track_id: DataFrame}
    """
    # get time average number
    number = [get_num_average_epoch(inc) for inc in data.sp_inc_angle]
    avg_vars = ["ddm_nbrcs", "ddm_les", "range_corr_gain"]
    avg_data = defaultdict(list)
    for i in range(len(data)):
        num = number[i]
        # data location at terminal or have interval
        if ((1 == num) or (i == (len(data)-1)) or
            (time_difference(data.ddm_timestamp_utc.iloc[i+1],
             data.ddm_timestamp_utc.iloc[i])) > 1.0):
            for var in avg_vars:
                avg_data[var].append(data[var].iloc[i])
            continue
        # time averaging
        ind = compute_average(i, num, data.ddm_timestamp_utc)
        for var in avg_vars:
            avg_data[var].append(np.mean(data[var].iloc[ind]))
    return avg_data


#########################################################################
def data_time_average(data):
    # #groupby with track_id
    data_dict = dict(list(data.groupby(data.sc_num)))
    avg_df = pd.DataFrame()
    for num in data_dict:
        # #groupby with track_id
        sp_data = data_dict[num]
        group_data = dict(list(sp_data.groupby(sp_data.track_id)))
        # #for each segmental arc
        for key in group_data:
            # #sort base on epoch time
            sorted_data = group_data[key].sort_values(by='ddm_timestamp_utc')
            # get DDMA, LES time averaging value for FDS wind speed retrieval
            avg_data = time_averaging(sorted_data)
            for var in avg_data:
                sorted_data[var] = avg_data[var]
            avg_df = avg_df.append(sorted_data)
    return avg_df


#########################################################################
def mix_day_cyg(path, file_list):
    """ mix each day single cygnss L1 file data to one dict, like l2 file """
    data = {}
    date = None
    for file in file_list:
        print(os.path.join(path, file))
        # define G1data object for data extration
        gr = GRData(os.path.join(path, file))
        if not gr.data:
            continue
        if date and (date != gr.data.date):  # confirm data at same day
            print('DataSet is not at same day, happend at{}'.format(path))
        date = gr.data.date
        # record needed data
        data[gr.data.attrs["sc_num"]] = gr.data
    return date, data


#########################################################################
def retrieval_prepare(file_manager, r_path):
    """
    saving each spacecraft data file to one single file,
    like cygnss l2 file
    """
    if not os.path.exists(r_path):
        os.makedirs(r_path)
    for path in file_manager:
        date, g_wind = mix_day_cyg(path, file_manager[path])
        if not g_wind:
            continue
        # data = data_specular_filter(data_combine(g_wind))  # compute ddma and les
        # data = compute_effective_area(data_combine(g_wind))
        data = data_combine(g_wind)   # use DDMA and LES
        if isinstance(date, datetype):
            filename = 'cyg.ddmi'+date.strftime('%Y%m%d')+'.p1.pkl'
            write_pickle(os.path.join(r_path, filename), (date, data))


#########################################################################
def match_wv(g_path, nwp_path, filename):
    """ match reference wind vector between ERA5 and DDM """
    file_list = os.listdir(g_path)
    if not file_list:
        return
    file_list = [file for file in file_list if file.endswith('.pkl')]
    nwp = ERA()
    match = Match_nwp()
    data = pd.DataFrame()
    for file in file_list:  # single day file
        print(file)
        date, g_wind = pd.read_pickle(os.path.join(g_path, file))
        if not nwp.pickfile(date, nwp_path):
            print('{} ERA data is missing, please confirm!'.format(date))
            continue
        nwp.readnc()
        match.matchup(g_wind, nwp)
        ta_data = data_time_average(match.data)
        data = data.append(ta_data)
    if data.empty:
        return
    write_pickle(filename, data)


#########################################################################
def reconstruct_data(g_path, filename):
    """ clear predict data for wind predict """
    file_list = os.listdir(g_path)
    if not file_list:
        return
    file_list = [file for file in file_list if file.endswith('.pkl')]
    data = pd.DataFrame()
    for file in file_list:  # single day file
        print(file)
        _, g_wind = pd.read_pickle(os.path.join(g_path, file))
        g_wind = g_wind.dropna("time")
        g_wind = g_wind.to_dataframe()
        ta_data = data_time_average(g_wind)
        data = data.append(ta_data)
    if data.empty:
        return
    write_pickle(filename, data)
