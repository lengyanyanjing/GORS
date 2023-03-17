import os
import numpy as np
import pandas as pd
import scipy.stats as scistats
import pickle
import multiprocessing as multiproc
from scipy.interpolate import interp1d
from utilities import df_combine

# =============================================================================
# This subrotine aims to mapping DDM observables to individual observables wind
# and estimate minimum variation coefficients for each DDM observales
# =============================================================================


###############################################################################
def load_data(obs):
    """ load parametric gmf files """
    if isinstance(obs, str):
        return pd.read_pickle(obs)
    # direct input data
    elif isinstance(obs, pd.DataFrame):
        return obs


###############################################################################
def split_data(length, n):
    """ average split list to n slices """
    step = int(np.floor(length*1./n))
    start = 0
    while (n > 0):
        n -= 1
        end = start+step
        if n == 0:
            yield (start, length)
            break
        yield (start, end)
        start += step


###############################################################################

def incidence_dim_interp_paras(obs_type, x, incidence_list, gmf):
    """
        interoplate the observables to inversed wind at each
        incidence angle interval, using parametric GMF,
        which is smoothed empirical GMF with linear interploation
    """
    if obs_type == 'ddm_nbrcs':
        var = 'paras_nbrcs'
    elif obs_type == 'ddm_les':
        var = 'paras_les'
    # interp_obs = np.zeros((len(gmf)))
    for angle in incidence_list:
        # 1D linear interpolate for nbrcs
        interpolant = interp1d(gmf[angle][var],
                               gmf[angle]['paras_wind'],
                            #    kind='quadratic',
                               kind='linear',
                               fill_value='extrapolate')
        yield interpolant(x)

        
###############################################################################
def incidence_dim_interp_emp(obs_type, x, incidence_list, gmf):
    """
        interoplate the observables inversed wind at each
        incidence angle interval, using un-smoothed GMF (empirical GMF)
    """
    if obs_type == 'ddm_nbrcs':
        var = 'emp_nbrcs'
    elif obs_type == 'ddm_les':
        var = 'emp_les'
    # interp_obs = np.zeros((len(gmf)))
    for angle in incidence_list:
        # 1D linear interpolate for nbrcs
        interpolant = interp1d(gmf[angle][var],
                               gmf[angle]['emp_wind'],
                            #    kind='quadratic',
                               kind='linear',
                               fill_value='extrapolate')
        yield interpolant(x)


###############################################################################
def observable_dim_interp(obs_type, data, index, incidence_list, gmf):
    data_df = data.iloc[index[0]:index[1], :]
    for row in data_df.itertuples(index=True, name='Pandas'):
        iter_interped = incidence_dim_interp_paras(obs_type,
                                                   getattr(row, obs_type),
                                                   incidence_list,
                                                   gmf)
        interped_array = [i for i in iter_interped]
        interpolant = interp1d(incidence_list, np.array(interped_array),
                               kind='linear', fill_value='extrapolate')
        yield interpolant(getattr(row, 'sp_inc_angle'))

    
###############################################################################
def wind_mapping(inputs):
    """ mapping DDM observables to corresponding inversed wind """
    data, index, nbrcs_gmf, les_gmf = inputs
    nbrcs_incidence_list = np.array(list(nbrcs_gmf.keys()))
    les_incidence_list = np.array(list(les_gmf.keys()))
    n_iter = observable_dim_interp('ddm_nbrcs', data, index,
                                   nbrcs_incidence_list, nbrcs_gmf)
    l_iter = observable_dim_interp('ddm_les', data, index,
                                   les_incidence_list, les_gmf)
    nbrcs_wind = [n_wind for n_wind in n_iter]
    les_wind = [l_wind for l_wind in l_iter]
    return {index: (nbrcs_wind, les_wind)}


###############################################################################
def save2file(data, filename):
    with open(filename, 'wb') as pfile:
        if isinstance(data, list):
            for x in data:
                pickle.dump(x, pfile, pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(data, pfile, pickle.HIGHEST_PROTOCOL)


###############################################################################
def get_stats(group):
    """ calculate the grouped mean, and std for MV """
    return {'nbrcs_bias': group.diff_nbrcs.mean(),
            'les_bias': group.diff_les.mean(),
            'nbrcs_std': group.diff_nbrcs.std(),
            'les_std': group.diff_les.std()}


###############################################################################
def obs_wind_retreival(obs_filename, gmf_filename, save_filename):
    """ one of main function for processing observables wind mapping"""
    nbrcs_gmf, les_gmf = load_data(gmf_filename)
    data_df = df_combine(load_data(obs_filename),
                         flag='all', type='retrieval')
    index = (0, len(data_df))
    wind = wind_mapping((data_df, index, nbrcs_gmf, les_gmf))
    data_df['nbrcs_wind'] = wind[index][0]
    data_df['les_wind'] = wind[index][1]
    data_df = data_df[(data_df.nbrcs_wind-data_df.les_wind).abs() <= 3.]
    save2file(data_df, save_filename)


###############################################################################
def multiproc_mapping(obs_filename, gmf_filename, save_filename):
    """  parallel computing the DDM inversed wind for each obserbles """
    nbrcs_gmf, les_gmf = load_data(gmf_filename)
    data = load_data(obs_filename)
    if isinstance(data, dict):
        data_df = df_combine(data, flag='all', type='retrieval')
    else:
        data_df = data
    itera = split_data(len(data_df), multiproc.cpu_count())
    inputs = []
    for index in itera:
        inputs.append((data_df, index, nbrcs_gmf, les_gmf))
    pool = multiproc.Pool(processes=multiproc.cpu_count()*2,
                          maxtasksperchild=2)
    result_list = pool.map(wind_mapping, inputs)
    pool.close()
    pool.join()
    pool.terminate()
    result = {}
    [result.update(i) for i in result_list]
    nbrcs_wind = []
    les_wind = []
    for index in sorted(list(result.keys())):
        nbrcs_wind.extend(result[index][0])
        les_wind.extend(result[index][1])
    data_df['nbrcs_wind'] = nbrcs_wind
    data_df['les_wind'] = les_wind
    # data filter, eliminate the NaN, inf and large difference
    data_df = data_df[(data_df.nbrcs_wind-data_df.les_wind).abs() <= 3.]
    # data_df = data_df.replace([np.Inf, -np.Inf], np.NaN)
    # data_df = data_df.dropna()
    save2file(data_df, save_filename)


###############################################################################
def split_file(num_files, datafile):
    """
        split data files to the specific number of files,
        solve the files to big
    """
    data_df = df_combine(load_data(datafile), flag='all', type='retrieval')
    itera = split_data(len(data_df), num_files)
    filename = os.path.basename(datafile)
    filepath = os.path.dirname(datafile)
    for i, index in enumerate(itera):
        tmp = os.path.splitext(filename)
        saved_name = "{}_{}{}".format(tmp[0], str(i), tmp[1])
        saved_name = os.path.join(filepath, saved_name)
        save2file(data_df[index[0]:index[1]], saved_name)
        yield saved_name


###############################################################################
def combine_file(file_list, save_filename):
    for ifile in file_list:
        try:
            data_df = data_df.append(load_data(ifile))
        except UnboundLocalError:
            data_df = load_data(ifile)
    if save_filename is not None:
        save2file(data_df, save_filename)


###############################################################################
def multproc_estimate(n, data_filename, gmf_filename):
    """
        For big data files, need to split to the number of n files, via
        multi-processing
    """
    file_list = split_file(n, data_filename)
    result_file_list = []
    filepath = os.path.dirname(data_filename)
    for ifile in file_list:
        tmp_name = os.path.basename(ifile)
        temp = os.path.splitext(tmp_name)
        filename = "{}_mp_tmp{}".format(temp[0], temp[1])
        filename = os.path.join(filepath, filename)
        print(filename)
        result_file_list.append(filename)
        multiproc_mapping(ifile, gmf_filename, filename)
    return result_file_list


###############################################################################
def mv_estimator(data='mpwind_df.pkl', filename='mv_coefs.pkl'):
    """
        input data has included observable mapping wind, which is pandas
        DataFrame format
    """
    # define RCG interval for MV estimator
    rcg_bins = [1, 3, 5, 10, 20, 30, 70, 110, 150, np.Inf]
    data_df = load_data(data)  # load data
    data_df = data_df[(data_df.nbrcs_wind-data_df.les_wind).abs() <= 3.]
    data_df['diff_nbrcs'] = data_df.nbrcs_wind-data_df.WS
    data_df['diff_les'] = data_df.les_wind-data_df.WS
    partitions = pd.cut(data_df.range_corr_gain, bins=rcg_bins, right=False)
    # compute the bias and unbias std for MV estimate
    grouped = data_df.groupby(partitions)
    stats = grouped.apply(get_stats).to_dict()
    group_dict = dict(list(grouped))
    I_mat = np.mat(np.ones((2, 1)))
    mv_coef = {}
    mv_std = {}
    for key in group_dict:
        # C = SRS
        data = group_dict[key]
        try:
            if data.empty:
                continue
            corr = scistats.pearsonr(data.diff_nbrcs, data.diff_les)
            R_mat = np.mat([[1., corr[0]], [corr[0], 1.]])
            S_mat = np.mat(np.diag([stats[key]['nbrcs_std'],
                                    stats[key]['les_std']]))
            C_mat_I = (S_mat*R_mat*S_mat).I
        except ValueError:
            continue
        # calculate the MV delta and MV weighting coefficient
        coef = 1./C_mat_I.sum()
        mv_std[key] = coef
        mv_coef[key] = coef*C_mat_I*I_mat
    save2file([mv_coef, mv_std], filename)


###############################################################################
if __name__ == '__main__':
    # =========================================================================
    # single process code
    #
    # obs_wind_retreival()
    #
    # =========================================================================

    # =========================================================================
    # #multiprocessing code for obs wind mapping

    gmf_filename = "pca_parametric_gmf.pkl"
    obs_filename = "pca_wd_test_dataset.pkl"
    save_filename = os.path.join(RESULT_PATH, "pca_wd_test_obs_wind.pkl")
    multiproc_mapping(obs_filename, gmf_filename, save_filename)

    # =========================================================================
    
    # =========================================================================
    # #for big data, split data to small files, then processing
    # gmf_filename = "pca_parametric_gmf.pkl"
    # data_filename = "pca_wd_train_dataset.pkl"
    # file_list = multproc_estimate(5, data_filename, gmf_filename)
    # save_filename = os.path.join(RESULT_PATH, "pca_wd_train_obs_wind.pkl")
    # combine_file(file_list, save_filename)
    #
    # =========================================================================

    # =========================================================================
    # #mv estimator to generate the mv coefficient
    #
    # mv_estimator("pca_wd_train_obs_wind.pkl", "pca_mv_coefs.pkl")
    #
    # =========================================================================
