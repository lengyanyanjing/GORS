import os
from gmf import EGMF_generator
from utilities import clear_dir
from smooth import paras_gmf
from pypre import (load_cyg_dataset, match_wv,
                   retrieval_prepare, reconstruct_data)
from estimator import (multproc_estimate, combine_file,
                       multiproc_mapping, mv_estimator)
from predictor import Predictor
from output import wind_evaluate


TEMP_PATH = r".\temp"
RESULT_PATH = r".\result"


# -----------------------------------------------------------------------------
def wind_speed_modeling(data_path, nwp_path, start_date, end_date,
                        gmf_file="parametric_gmf.pkl", mv_file="mv_coefs.pkl",
                        tmp_path=TEMP_PATH, result_path=RESULT_PATH):
    """ """
    # # set temporary path and result path
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    clear_dir(tmp_path)
    # # (1)get file and dir as dict, data match up
    file_mag = load_cyg_dataset(data_path, 'date', start_date, end_date)
    retrieval_prepare(file_mag, tmp_path)
    train_filename = os.path.join(result_path, "train_dataset.pkl")
    match_wv(tmp_path, nwp_path, train_filename)
    # # (2)generate emperical 2D GMF
    emp_gmf = EGMF_generator(train_filename)
    emp_gmf.gmf_soc3()
    emp_gmf_filename = os.path.join(result_path, "empirical_gmf.pkl")
    emp_gmf.save2file(emp_gmf_filename)
    # # (3)emperical 2D GMF smooth
    paras_gmf(emp_gmf_filename, os.path.join(result_path, gmf_file))
    # # (4)mapping wind of DDM observales
    obs_wind_filename = os.path.join(result_path, "train_obs_wind.pkl")
    file_list = multproc_estimate(5, train_filename,
                                  os.path.join(result_path, gmf_file))
    combine_file(file_list, obs_wind_filename)
    # # (5)mv estimate the weighting coefficents
    mv_estimator(obs_wind_filename, os.path.join(result_path, mv_file))


# -----------------------------------------------------------------------------
def wind_speed_predict(data_path, start_date, end_date,
                       filename="predict_ws.pkl",
                       gmf_file="parametric_gmf.pkl", mv_file="mv_coefs.pkl",
                       tmp_path=TEMP_PATH, result_path=RESULT_PATH):
    # # set temporary path and result path
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    clear_dir(tmp_path)
    # # (1)get file and dir as dict, data clear
    file_mag = load_cyg_dataset(data_path, 'date', start_date, end_date)
    retrieval_prepare(file_mag, tmp_path)
    test_filename = os.path.join(result_path, "predict_dataset.pkl")
    reconstruct_data(tmp_path, test_filename)
    # # (2)data predict
    obs_wind_filename = os.path.join(result_path, "predict_obs_wind.pkl")
    multiproc_mapping(test_filename, os.path.join(result_path, gmf_file),
                      obs_wind_filename)
    wp_predict = Predictor(os.path.join(result_path, mv_file))
    wp_predict.predictor(obs_wind_filename)
    wp_predict.saveData(os.path.join(result_path, filename))


# -----------------------------------------------------------------------------
def wind_speed_modeling_test(data_path, nwp_path, start_date, end_date,
                             filename="train_final_ws.pkl",
                             gmf_file="parametric_gmf.pkl", mv_file="mv_coefs.pkl",
                             tmp_path=TEMP_PATH, result_path=RESULT_PATH):
    """ """
    # # set temporary path and result path
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    clear_dir(tmp_path)
    # # (1)get file and dir as dict, data match up
    file_mag = load_cyg_dataset(data_path, 'date', start_date, end_date)
    retrieval_prepare(file_mag, tmp_path)
    train_filename = os.path.join(result_path, "train_dataset.pkl")
    match_wv(tmp_path, nwp_path, train_filename)
    # # (2)generate emperical 2D GMF
    emp_gmf = EGMF_generator(train_filename)
    emp_gmf.gmf_soc3()
    emp_gmf_filename = os.path.join(result_path, "empirical_gmf.pkl")
    emp_gmf.save2file(emp_gmf_filename)
    # # (3)emperical 2D GMF smooth
    paras_gmf(emp_gmf_filename, os.path.join(result_path, gmf_file))
    # # (4)mapping wind of DDM observales
    obs_wind_filename = os.path.join(result_path, "train_obs_wind.pkl")
    file_list = multproc_estimate(5, train_filename,
                                  os.path.join(result_path, gmf_file))
    combine_file(file_list, obs_wind_filename)
    # # (5)mv estimate the weighting coefficents
    mv_estimator(obs_wind_filename, os.path.join(result_path, mv_file))
    # # (6)retrieval final wind speed
    wp_predict = Predictor(os.path.join(result_path, mv_file))
    wp_predict.predictor(obs_wind_filename)
    filename = os.path.join(result_path, filename)
    wp_predict.saveData(filename)
    wind_evaluate(filename)


# -----------------------------------------------------------------------------
def wind_speed_predict_test(data_path, nwp_path, start_date, end_date,
                            filename="predict_wp.pkl",
                            gmf_file="parametric_gmf.pkl", mv_file="mv_coefs.pkl",
                            tmp_path=TEMP_PATH, result_path=RESULT_PATH):
    # # set temporary path and result path
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    clear_dir(tmp_path)
    # # (1)get file and dir as dict, data clear
    file_mag = load_cyg_dataset(data_path, 'date', start_date, end_date)
    retrieval_prepare(file_mag, tmp_path)
    test_filename = os.path.join(result_path, "predict_dataset.pkl")
    match_wv(tmp_path, nwp_path, test_filename)
    # # (2)data predict
    obs_wind_filename = os.path.join(result_path, "predict_obs_wind.pkl")
    multiproc_mapping(test_filename, os.path.join(result_path, gmf_file),
                      obs_wind_filename)
    wp_predict = Predictor(os.path.join(result_path, mv_file))
    wp_predict.predictor(obs_wind_filename)
    filename = os.path.join(result_path, filename)
    wp_predict.saveData(filename)
    wind_evaluate(filename)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    data_path = r"F:\CYGNSS L1"
    nwp_path = r"F:\ERA5\atmosphere"
    start_date = "2017/08/01"
    end_date = "2017/08/05"
    # wind_speed_modeling(data_path, nwp_path, start_date, end_date)
    # wind_speed_predict(data_path, start_date, end_date)
    # wind_speed_predict_test(data_path, nwp_path, start_date, end_date)
