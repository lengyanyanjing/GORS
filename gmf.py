#!usr/bin/evn/win/python 3.6, by DZN, 2018/10/31
import os
import pickle
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from utilities import df_combine


"""
    This module designing is for realize the GNSS-R wind speed retrieval
    algorithm. The input data is from L1 temperary files from preparing
    retrieval module (pretri.py).
"""


#########################################################################
class EGMF_generator(object):
    """ this class is design for compute the empirical GMF of observables """
    def __init__(self, filename):
        """
            Read matchup paris for empirical GMF derive, the input file from
            above functions recording as pickle format file with true wind
        """
        if filename and os.path.isfile(filename):
            data = pd.read_pickle(filename)
            # filter the observables
            data = data[data.ddm_nbrcs > 0.0]
            data = data[data.ddm_les > 0.0]
            data = data[data.sp_inc_angle < 68.0]
            self.data = data[data.range_corr_gain > 10.0]
            
        # save empirical gmf {incidence:{'nbrcs':[], 'wind':[]}}
        self.gmf_nbrcs = {}   # scatter for empirical GMF of nbrcs
        self.gmf_les = {}     # scatter for empirical GMF of les
        self.wind_lim = (0., 35.)       # bin center of wind speed
        self.indicence_lim = (.5, 70.5)  # bin center of incidence angle
        self.incd_bin_width = 2.  # bin width of incidence angle

    def _get_index(self, center, bin_width, data):
        # determine +/- one bin width sample
        lower_lim = center-bin_width
        upper_lim = center+bin_width
        ind = ((data >= lower_lim) & (data < upper_lim))
        # determin sample between 1 bin width and 2 bin width
        ex_lower_lim = lower_lim-bin_width
        ex_upper_lim = upper_lim+bin_width
        ex_ind = np.logical_or(((data >= ex_lower_lim) &
                               (data < lower_lim)),
                               ((data >= upper_lim) &
                               (data < ex_upper_lim)))
        return ind, ex_ind

    def _weighted_avg(self, wind_center, wind_bin_width, data, data_ex):
        """
            this function aim at calculate the weighting value of
            true wind and observables at each bins
        """
        vars = ['WS', 'ddm_nbrcs', 'ddm_les']
        incd_weight_ind = incd_weight_ex = .0
        # close bin incidence center and between 1bin and 2bin have sample
        if (not data.empty) and (not data_ex.empty):
            incd_weight_ind = 2.
            incd_weight_ex = 1.
        # sample just center on 1bin of incidence
        elif np.all(data_ex.empty):
            incd_weight_ind = 1.
        # bin width for true wind dimention
        ind1, ex_ind1 = self._get_index(wind_center,
                                        wind_bin_width, data.WS)
        try:
            ind2, ex_ind2 = self._get_index(wind_center,
                                            wind_bin_width, data_ex.WS)
        except AttributeError:
            ind2 = ex_ind2 = np.array([False])
            pass
        # bin width for observables bins
        var_weight_ind = var_weight_ex = .0
        if np.logical_and(np.any(np.hstack((ind1, ind2))),
                          np.any(np.hstack((ex_ind1, ex_ind2)))):
            var_weight_ind = 2.
            var_weight_ex = 1.
        elif np.all(np.logical_not(np.hstack((ex_ind1, ex_ind2)))):
            var_weight_ind = 1.
        # total weight values
        total_weight = np.count_nonzero(ind1)*var_weight_ind*incd_weight_ind
        total_weight += np.count_nonzero(ind2)*var_weight_ind*incd_weight_ex
        total_weight += np.count_nonzero(ex_ind1)*var_weight_ex*incd_weight_ind
        total_weight += np.count_nonzero(ex_ind2)*var_weight_ex*incd_weight_ex
        # data empty at this bin
        if total_weight == 0.0:
            return [np.NaN]*len(vars)
        avg = []
        #        _ _ _ weighting strategy for overlap data
        #       |_|_|_|                         1 2 1
        #       |_|_|_| <- incidence angle  ->  2 4 2     
        #       |_|_|_|                         1 2 1  
        #          ^                              ^
        #          |                              |
        #         wind                           wind
        # calculatet the true wind, observables weighted values
        for var in vars:
            temp = np.sum(data.loc[ind1, var].values *
                          (var_weight_ind*incd_weight_ind/total_weight))
            temp += np.sum(data.loc[ex_ind1, var].values *
                           (var_weight_ex*incd_weight_ind/total_weight))
            if not data_ex.empty:
                temp += np.sum(data_ex.loc[ind2, var].values *
                               (var_weight_ind*incd_weight_ex/total_weight))
                temp += np.sum(data_ex.loc[ex_ind2, var].values *
                               (var_weight_ex*incd_weight_ex/total_weight))
            avg.append(temp)
        return avg
  
    def _gmf(self, incidence, grouped):
        wind = grouped.WS.mean().values
        nbrcs = grouped.ddm_nbrcs.mean().values
        les = grouped.ddm_les.mean().values
        # empirial gmf with normal average
        self.gmf_nbrcs[incidence] = {'wind': wind, 'nbrcs': nbrcs}
        self.gmf_les[incidence] = {'wind': wind, 'les': les}

    def _gmf_1(self, incidence, grouped):
        # start the bin from center wind equal to 7.05m/s,
        # the PDF of this bin is maximum
        wpdf_max = 7  # corresponding to the bins centered on 7.05 m/s in the WP demision
        grouped_dict = dict(list(grouped))
        wind_list = sorted(list(grouped_dict.keys()))
        w_avge = grouped_dict[wpdf_max].WS.mean()
        b_avge = grouped_dict[wpdf_max].ddm_nbrcs.mean()
        l_avge = grouped_dict[wpdf_max].ddm_les.mean()
        wind = [w_avge]
        nbrcs = [b_avge]
        les = [l_avge]
        # wind < 7.05 m/s
        for i in reversed(wind_list[:wind_list.index(wpdf_max)]):
            wind = [grouped_dict[i].WS.mean()]+wind
            temp_nbrcs = grouped_dict[i].ddm_nbrcs.mean()
            temp_les = grouped_dict[i].ddm_les.mean()
            if temp_nbrcs < nbrcs[0]:
                temp_nbrcs = nbrcs[0]
            nbrcs = [temp_nbrcs]+nbrcs
            if temp_les < les[0]:
                temp_les = les[0]
            les = [temp_les]+les
        # wind > 7.05 m/s
        for i in wind_list[wind_list.index(wpdf_max)+1:]:
            wind.append(grouped_dict[i].WS.mean())
            temp_nbrcs = grouped_dict[i].ddm_nbrcs.mean()
            temp_les = grouped_dict[i].ddm_les.mean()
            temp_nbrcs = nbrcs[-1] if temp_nbrcs > nbrcs[-1] else temp_nbrcs
            nbrcs.append(temp_nbrcs)
            temp_les = les[-1] if temp_les > les[-1] else temp_les
            les.append(temp_les)
        wind = np.array(wind)
        nbrcs = np.array(nbrcs)
        les = np.array(les)  
        # empirial gmf with normal average
        self.gmf_nbrcs[incidence] = {'wind': wind, 'nbrcs': nbrcs}
        self.gmf_les[incidence] = {'wind': wind, 'les': les}

    def _gmf_2(self, incidence, data, data_ex, start_center):
        # start the bin from center wind equal to 7.05m/s,
        # the PDF of this bin is maximum
        # # wind = 7.05
        wind_bin_width = self._get_wind_binwidth(start_center)
        avg = self._weighted_avg(start_center, wind_bin_width,
                                 data, data_ex)
        wind, nbrcs, les = [avg[0]], [avg[1]], [avg[2]]
        # # wind < 7.05 m/s
        for w in reversed(np.arange(0.05, start_center, 0.1)):
            bin_width = self._get_wind_binwidth(w)
            avg = self._weighted_avg(w, bin_width, data, data_ex)
            wind = [avg[0]]+wind
            if avg[1] < nbrcs[0]:
                avg[1] = nbrcs[0]
            nbrcs = [avg[1]]+nbrcs
            if avg[2] < les[0]:
                avg[2] = les[0]
            les = [avg[2]]+les
        # # wind > 7.05 m/s
        for w in np.arange(start_center+0.1, 35., 0.1):
            bin_width = self._get_wind_binwidth(w)
            avg = self._weighted_avg(w, bin_width, data, data_ex)
            wind.append(avg[0])
            avg[1] = nbrcs[-1] if avg[1] > nbrcs[-1] else avg[1]
            nbrcs.append(avg[1])
            avg[2] = les[-1] if avg[2] > les[-1] else avg[2]
            les.append(avg[2])
        wind = np.array(wind)
        nbrcs = np.array(nbrcs)
        les = np.array(les)  
        # empirial gmf with normal average
        if incidence:
            self.gmf_nbrcs[incidence] = {'wind': wind, 'nbrcs': nbrcs}
            self.gmf_les[incidence] = {'wind': wind, 'les': les}
        else:
            self.gmf_nbrcs = {'wind': wind, 'nbrcs': nbrcs}
            self.gmf_les = {'wind': wind, 'les': les}

    def _get_wind_binwidth(self, wind):
        """ determine the binwidth of wind speed """
        wind_discontinu = [2, 5, 9, 11, 14, 17]
        wind_binwidth = [.4, .3, .2, .4, .6, .8, 1.0]
        for i, w in enumerate(wind_discontinu):
            if wind <= w:
                return wind_binwidth[i]
        return wind_binwidth[-1]

    def load_retrieval_data(self, filename=None):
        if os.path.isfile(filename):
            self.data = df_combine(pd.read_pickle(filename),
                                     flag='all', type='retrieval')
            self.data = self.data[self.data.ddm_nbrcs > 0.]
            self.data = self.data[self.data.ddm_les > 0.]
            return self.data
        elif filename is None:
            return self.data

    def gmf_soc0(self):
        """ 
        empirical GMF same like Soc of cygnss,
        but directly calculate the average in the uniformed bins
        """
        # limit the incidence angle in the range of 1~70 degree
        self.data = self.data[self.data.sp_inc_angle < self.indicence_lim[1]]
        # set the incidence angle step as 1 degree from 1 to 70 degree
        self.data['incidence_cat'] = np.round(self.data['sp_inc_angle'])
        # set the wind step center as 0.1m/s from 0.05 to 34.5m/s
        interval_index = np.arange(self.wind_lim[0], self.wind_lim[1], 0.1)
        self.data['wind_cat'] = pd.cut(self.data.WS, bins=interval_index)
        for angle, incidence_group in self.data.groupby('incidence_cat'):
            self._gmf(angle, incidence_group.groupby('wind_cat'))

    def gmf_soc1(self):
        # limit the incidence angle in the range of 1~70 degree
        self.data = self.data[self.data.sp_inc_angle < self.indicence_lim[1]]
        # set the incidence angle step as 1 degree from 1 to 70 degree
        self.data['incidence_cat'] = np.round(self.data['sp_inc_angle'])
        # set the wind step as 0.1m/s from 0.05 to 34.5m/s
        interval_index = np.arange(self.wind_lim[0], self.wind_lim[1], 0.1)
        self.data['wind_cat'] = pd.cut(self.data.WS, bins=interval_index)
        grouped_dict = dict(list(self.data.groupby('incidence_cat')))
        incidence = sorted(list(grouped_dict.keys()))
        for i, angle in enumerate(incidence):
            incidence_group = grouped_dict[angle]
            for j in (i-1, i+1):
                if (j < 0) or (j >= len(grouped_dict)):
                    continue
                incidence_group = incidence_group.append(grouped_dict[incidence[j]])
            self._gmf(angle, incidence_group.groupby('wind_cat'))
    
    def gmf_soc2(self):
        # limit the incidence angle in the range of 1~70 degree
        self.data = self.data[self.data.sp_inc_angle < self.indicence_lim[1]]
        # set the incidence angle step as 1 degree from 1 to 70 degree
        self.data['incidence_cat'] = np.round(self.data['sp_inc_angle'])
        # set the wind step as 0.1m/s from 0.05 to 34.5m/s
        interval_index = np.arange(self.wind_lim[0], self.wind_lim[1], 0.1)
        self.data['wind_cat'] = pd.cut(self.data.WS,
                                       bins=interval_index,
                                       labels=False)
        grouped_dict = dict(list(self.data.groupby('incidence_cat')))
        for angle in sorted(list(grouped_dict.keys())):
            lower_lim = angle-2.
            upper_lim = angle+2.
            ind = ((self.data.sp_inc_angle >= lower_lim) &
                   (self.data.sp_inc_angle < upper_lim))
            incidence_group = self.data[ind]   
            self._gmf_1(angle, incidence_group.groupby('wind_cat'))
            
    def gmf_soc3(self):
        """ final empirical GMF calculation funstion as Ruf mentioned """
        # define incidence angle bin center
        incd_bin_center = np.arange(1., 71.)
        bin_list = np.arange(0.05, 35., 0.1)
        for angle in incd_bin_center:
            ind, ex_ind = self._get_index(angle, self.incd_bin_width,
                                          self.data.sp_inc_angle)
            # get bin center data and close interval data
            data = self.data[ind]
            data_ex = self.data[ex_ind]
            hist, _ = np.histogram(self.data[(ind | ex_ind)].WS,
                                   bins=len(bin_list))
            start_center = bin_list[hist.argmax()]
            # calculate empirical gmf at specific incidence angle
            self._gmf_2(angle, data, data_ex, start_center)
            print('incidence: {}, maximum bins of true wind: {}'.format(angle, hist.argmax()))

    def gmf_buoy(self):
        """ final empirical GMF calculation funstion as Ruf mentioned for buoy"""
        # define incidence angle bin center
        bin_list = np.arange(0.05, 35., 0.1)
        hist, _ = np.histogram(self.data.WS, bins=len(bin_list))
        start_center = bin_list[hist.argmax()]
        self._gmf_2(None, self.data, pd.DataFrame(), start_center)
    
    def scatter_matrix(self):
        """ plot the scatterplot for specific features """
        attributions = ['sp_inc_angle', 'range_corr_gain',
                        'ddm_nbrcs', 'ddm_les', 'WS']
        pd.plotting.scatter_matrix(self.data[attributions], figsize=(12, 8))

    def plotting_gmf(self):
        """ plotting the empirical GMF at incidence angle list """
        incidence_list = [10., 15., 30., 45., 50., 55.]
        fig, ax = plt.subplots(2, 1, figsize=(5, 7.5))
        for angle in incidence_list:
            if angle not in self.gmf_nbrcs:
                continue
            ax[0].plot(self.gmf_nbrcs[angle]['wind'],
                       self.gmf_nbrcs[angle]['nbrcs'],
                       label='{}'.format(angle))
            ax[1].plot(self.gmf_les[angle]['wind'],
                       self.gmf_les[angle]['les'],
                       label='{}'.format(angle))
        ax[0].set_xlabel('wind speed (m/s)')
        ax[1].set_xlabel('wind speed (m/s)')

        ax[0].set_title('DDMA FDS GMF (m/s)')
        ax[1].set_title('LES FDS GMF (m/s)')
        ax[0].grid(axis='y')
        ax[1].grid(axis='y')
        ax[0].legend(loc='best')
        ax[1].legend(loc='best')
        plt.tight_layout()

    def save2file(self, filename):
        with open(filename, 'wb') as pfile:
            pickle.dump((self.gmf_nbrcs, self.gmf_les),
                        pfile, pickle.HIGHEST_PROTOCOL)

    def savemat(self, filename):
        """ save variables as mat file for matlab using """
        sio.savemat(filename, self.data.to_dict('list'))


#########################################################################
def plotting_gmf(filename, flag=True):
    """ plot the empirical gmf from file """
    with open(filename, 'rb') as pfile:
        gmf_nbrcs, gmf_les = pickle.load(pfile)
        fig, ax = plt.subplots(2, 1, figsize=(5, 7.5))
        if flag:
            incidence_list = [10., 15., 30., 45., 50., 55.]
            for angle in incidence_list:
                if (angle not in gmf_nbrcs) or (angle not in gmf_les):
                    continue
                ax[0].plot(gmf_nbrcs[angle]['wind'],
                           gmf_nbrcs[angle]['nbrcs'],
                           label='{}'.format(angle))
                ax[1].plot(gmf_les[angle]['wind'],
                           gmf_les[angle]['les'],
                           label='{}'.format(angle))
        else:
            ax[0].plot(gmf_nbrcs['wind'],
                       gmf_nbrcs['nbrcs'],
                       label='NBRCS')
            ax[1].plot(gmf_les['wind'],
                       gmf_les['les'],
                       label='LES')
        ax[0].set_xlabel('wind speed (m/s)')
        ax[1].set_xlabel('wind speed (m/s)')
        ax[0].set_ylabel('DDMA')
        ax[1].set_ylabel('LES')
        ax[0].set_ylim(0, 250)
        ax[1].set_ylim(0, 100)
        ax[0].set_title('DDMA FDS GMF (m/s)')
        ax[1].set_title('LES FDS GMF (m/s)')
        ax[0].grid(axis='y')
        ax[1].grid(axis='y')
        ax[0].legend(loc='best')
        ax[1].legend(loc='best')
        fig.tight_layout()
        plt.show()
