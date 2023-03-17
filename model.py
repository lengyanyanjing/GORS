# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:51:50 2019

@author: DZN
"""
import os
import numpy as np
import pandas as pd
from utilities import data_combine
from sklearn.decomposition import PCA
from sklearn.externals import joblib


# =============================================================================
def extract_ddm_volums(power, delay_row, dopp_col, win_size=(3, 5)):
    """
    extract the subddm around the specular point
    delay_row : the specular point at the delay row index
    doppler_col: the specular point at the doppler column index 
    win_size: the size of windown
    """
    rows = delay_row.round().astype(int)
    cols = dopp_col.round().astype(int)-win_size[1]//2
    ind_delay = rows.copy()
    ind_dopp = cols.copy()
    for i in range(1, win_size[0]):
        ind_delay = np.vstack((ind_delay, rows+i))
    for i in range(1, win_size[1]):
        ind_dopp = np.vstack((ind_dopp, cols+i))
    shape = power.shape
    # # indexing layer, the first dimension
    ind0 = np.repeat(np.arange(shape[0], dtype=int), win_size[0]*win_size[1])
    ind0 = ind0.reshape(shape[0], win_size[0], win_size[1])
    # # indexing rows of the matrix
    ind1 = np.tile(ind_delay, (win_size[1], 1, 1))
    ind1 = ind1.transpose((2, 1, 0))
    # # indexing cols of the matrix
    ind2 = np.moveaxis(np.tile(ind_dopp, (win_size[0], 1, 1)), -1, 0)
    subpower = power[ind0, ind1, ind2]
    return subpower


# =============================================================================
def extract_waveform(power, dopp_col=None, flg=False):
    """
    extract delay waveform
    doppler: for the zero doppler determine
    flg: indicate the zero doppler if false, or means integrated waveform
    """
    if flg:
        waveform = power.sum(axis=2)
    else:
        if dopp_col is None:
            return
        cols = dopp_col.round().astype(int)
        shape = power.shape
        ind0 = np.repeat(np.arange(shape[0], dtype=int), shape[1])
        ind0 = ind0.reshape(shape[:2])
        ind1 = np.tile(np.arange(shape[1], dtype=int), (shape[0], 1))
        ind2 = np.repeat(cols, shape[1])
        ind2 = ind2.reshape(shape[:2])
        waveform = power[ind0, ind1, ind2]
    return waveform


# =============================================================================
def extract_ddm_power(power, delay_row, dopp_col):
    """ extract specular point ddm power or ddm peak power """
    rows = delay_row.round().astype(int)
    cols = dopp_col.round().astype(int)
    peak_power = power[np.arange(power.shape[0], dtype=int), rows, cols]
    return peak_power


# =============================================================================
def pca_filter(data):
    """ using fit PCA model for PCA analysis """
    # # shift to 2D array for sklearn
    shape = data.shape
    data = data.reshape((shape[0], shape[1]*shape[2]))
    pca = PCA(n_components=5).fit(data)
    components = pca.transform(data)
    filtered = pca.inverse_transform(components)
    return filtered.reshape(shape)


# =============================================================================
def specular_statistic(data):
    delay_row = data.brcs_ddm_sp_bin_delay_row.values.round().astype(int)
    dopp_col = data.brcs_ddm_sp_bin_dopp_col.values.round().astype(int)
    # row_count = np.array(np.unique(delay_row, return_counts=True)).T
    # col_count = np.array(np.unique(dopp_col, return_counts=True)).T
    unique_row_data, row_count = np.unique(delay_row, return_counts=True)
    max_row_data = unique_row_data[row_count.argmax()]
    unique_col_data, col_count = np.unique(dopp_col[delay_row == max_row_data],
                                           return_counts=True)
    max_col_data = unique_col_data[col_count.argmax()]
    print(max_row_data)
    print(max_col_data)
    print(col_count.max()/len(delay_row))


# =============================================================================
def compute_effective_area(data):
    """ re-compute sigma_ddma and sigma_les for filtered ddm """
    # # compute sigma_ddma
    sp_row, sp_col = 6, 5  # specify specular point in ddm
    delay_row = data.brcs_ddm_sp_bin_delay_row.values.round().astype(int)
    dopp_col = data.brcs_ddm_sp_bin_dopp_col.values.round().astype(int)
    index_delay = delay_row == sp_row
    index_dopp = dopp_col == sp_col
    index = index_delay & index_dopp
    data = data.isel(time=index)
    delay_row = delay_row[index]
    dopp_col = dopp_col[index]

    brcs = np.moveaxis(data.brcs.values, -1, 0)
    filtered_brcs = pca_filter(brcs)
    sub_brcs = extract_ddm_volums(filtered_brcs, delay_row, dopp_col)
    sigma_ddma = sub_brcs.sum(axis=(1, 2))

    sub_raw_brcs = extract_ddm_volums(brcs, delay_row, dopp_col)
    sigma_raw_ddma = sub_raw_brcs.sum(axis=(1, 2))
    ddma_raw_eff_area = sigma_raw_ddma/data.ddm_nbrcs.values
    filtered_ddma = sigma_ddma/ddma_raw_eff_area

    # # compute simga_les
    n = 3.0
    delay_chips = np.array([-0.25, 0.0, 0.25])
    divisor = sum(delay_chips**2)*n-(sum(delay_chips))**2
    sub_les = extract_ddm_volums(filtered_brcs, delay_row, dopp_col)
    sub_idw = sub_les.sum(axis=2)
    sigma_les = sub_idw.dot(delay_chips.T)*n-sum(delay_chips)*sub_idw.sum(axis=1)
    sigma_les /= divisor
    filtered_les = sigma_les/ddma_raw_eff_area
    # print(filtered_les[:10])
    # print(data.ddm_les.values[:10])

    data = data.drop(["brcs_ddm_sp_bin_delay_row",
                      "brcs_ddm_sp_bin_dopp_col",
                      "nbrcs_scatter_area", "les_scatter_area",
                      "quality_flags",
                      "power_analog", "brcs"])
    # data = data.to_dataframe()
    data["filtered_ddma"] = ("time", filtered_ddma)
    data["filtered_les"] = ("time", filtered_les)
    # print(data.columns)
    return data


# =============================================================================
def data_specular_filter(data):
    sp_row, sp_col = 6, 5  # specify specular point in ddm
    delay_row = data.brcs_ddm_sp_bin_delay_row.values.round().astype(int)
    dopp_col = data.brcs_ddm_sp_bin_dopp_col.values.round().astype(int)
    index_delay = delay_row == sp_row
    index_dopp = dopp_col == sp_col
    index = index_delay & index_dopp
    data = data.isel(time=index)
    # # PCA fit processing
    # power = np.moveaxis(data.power_analog.values, -1, 0)
    brcs = np.moveaxis(data.brcs.values, -1, 0)
    delay_row = delay_row[index]
    dopp_col = dopp_col[index]
    # plot_ddms(power)
    # plot_ddms(brcs)
    filtered_brcs = pca_filter(brcs)
    # filtered_power = pca_filter(power)
    # plot_ddms(filtered_power)
    # plot_ddms(filtered_brcs)
    sub_brcs = extract_ddm_volums(filtered_brcs, delay_row, dopp_col)
    sigma_ddma = sub_brcs.sum(axis=(1, 2))
    filtered_ddma = sigma_ddma/data.nbrcs_scatter_area.values

    n = 3.0
    delay_chips = np.array([-0.25, 0.0, 0.25])
    divisor = sum(delay_chips**2)*n-(sum(delay_chips))**2
    sub_les = extract_ddm_volums(filtered_brcs, delay_row, dopp_col)
    sub_idw = sub_les.sum(axis=2)
    sigma_les = sub_idw.dot(delay_chips)*n-sum(delay_chips)*sub_idw.sum(axis=1)
    sigma_les /= divisor
    filtered_les = sigma_les/data.les_scatter_area.values
    data = data.drop(["brcs_ddm_sp_bin_delay_row",
                      "brcs_ddm_sp_bin_dopp_col",
                      "nbrcs_scatter_area", "les_scatter_area",
                      "quality_flags",
                      "power_analog", "brcs"])
    # data = data.to_dataframe()
    data["filtered_ddma"] = ("time", filtered_ddma)
    data["filtered_les"] = ("time", filtered_les)
    return data
