import os
import pickle
import numpy as np
import pandas as pd
from estimator import load_data

RESULT_PATH = r'./result'               # result directory
if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)


###############################################################################
class Predictor:
    def __init__(self, mv_coef_filename):
        with open(mv_coef_filename, 'rb') as pfile:
            self.mv_coef = pickle.load(pfile)
            self.mv_std = pickle.load(pfile)

    def predictor(self, obs):
        """
            prdictor for calculating the final wind speed from GNSS-R,
            and filter the difference between nbrcs and les wind
        """
        self.data = load_data(obs)
        self.raw_data_length = len(self.data)
        # dividing the RCG interval
        rcg_bins = [1, 3, 5, 10, 20, 30, 70, 110, 150, np.Inf]
        partitions = pd.cut(self.data.range_corr_gain,
                            bins=rcg_bins, right=False)
        # groupby the data in different RCG interval
        group_dict = dict(list(self.data.groupby(partitions)))
        state = True
        for key in group_dict:
            try:
                coef = self.mv_coef[key]
            except KeyError:
                continue
            # filter the out-range difference between DDMA and LES wind
            data = group_dict[key]
            ind = abs(data.nbrcs_wind - data.les_wind) <= 3.
            data = data[ind]
            # computer the final wind speed using m*u
            data['wind'] = (data.nbrcs_wind*coef[0, 0] +
                            data.les_wind*coef[1, 0]).copy()
            if state:
                self.data = data
                state = False
            else:
                self.data = self.data.append(data)
        # record the data length
        self.final_data_length = len(self.data)

    def saveData(self, filename):
        with open(filename, 'wb') as pfile:
            pickle.dump(self.data, pfile, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    mv_coef_filename = "pca_mv_coefs.pkl"
    retri_wind_data = "pca_wd_test_obs_wind.pkl"
    wp_predict = Predictor(mv_coef_filename)
    wp_predict.predictor(retri_wind_data)
    saved_filename = "pca_test_final_wind.pkl"
    wp_predict.saveData(saved_filename)
