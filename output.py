import pandas as pd


# =============================================================================
def trim_data(data, vars_list=["WS", "prn_code",
                               "ddm_nbrcs", "ddm_les", 
                               "nbrcs_wind", "les_wind"]):
    # #drop the un-neccessay columns of DataFrame
    data = data.drop(vars_list, axis=1)
    data["wind_speed"] = data["wind"].astype('float64').copy()
    return data.drop("wind", axis=1)


###############################################################################
def get_stats(group):
    """ calculate the grouped mean, and std for MV """
    try:
        return {'bias': (group.wind-group.WS).mean(),
                'std': (group.wind-group.WS).std()}
    except AttributeError:
        return {'bias': (group.wind_speed-group.WS).mean(),
                'std': (group.wind_speed-group.WS).std()}


###############################################################################
def wind_evaluate(filename):
    """
        evaluate the accuracy of retrieval wind direct from DDM observables
    """
    data = pd.read_pickle(filename)
    data = data[data.sp_inc_angle <= 68]
    data = data[data.range_corr_gain >= 10]
    print("Data lenght {}".format(len(data)))
# =============================================================================
#     nbrcs_bias = (data.nbrcs_wind-data.WS).mean()
#     nbrcs_std = (data.nbrcs_wind-data.WS).std()
#     les_bias = (data.les_wind-data.WS).mean()
#     les_std = (data.les_wind-data.WS).std()
#     print("NBRCS wind error bias: {}".format(nbrcs_bias))
#     print("NBRCS wind error STD: {}".format(nbrcs_std))
#     print("LES wind error bias: {}".format(les_bias))
#     print("LES wind error STD: {}".format(les_std))
# =============================================================================
    if "wind_speed" in data.columns:
        """ wind_speed evaluates the L2 wind speed from L2 files """
        temp = (data.wind_speed-data.WS)
        print("CYGNSS wind error bias: {}".format(temp.mean()))
        print("CYGNSS wind error STD: {}".format(temp.std()))
        temp = data[data['WS'] < 20]
        temp = temp.wind_speed-temp.WS
        print("Wind speed less than 20 m/s bias {:.4f}".format(temp.mean()))
        print("Wind speed less than 20 m/s STD: {}".format(temp.std()))
        for i in [10., 15.]:
            midwp = data[data['WS'] > i]
            perc = len(midwp)*1./len(data)
            midstd = (midwp.wind_speed-midwp.WS).std()
            print("Wind speed larger than {} m/s account for {:.4f}%".format(i, perc*100))
            print("Wind speed larger than {} m/s STD: {}".format(i, midstd))
    if "wind" in data.columns:
        """ wind means the final retrieval wind speed from observables """
        temp = data.wind-data.WS
        print("Final wind error bias: {}".format(temp.mean()))
        print("Final wind error STD: {}".format(temp.std()))
        temp = data[data['WS'] < 20]
        temp = temp.wind-temp.WS
        print("Wind speed less than 20 m/s bias {:.4f}".format(temp.mean()))
        print("Wind speed less than 20 m/s STD: {}".format(temp.std()))
        for i in [10., 15.]:
            midwp = data[data['WS'] > i]
            perc = len(midwp)*1./len(data)
            midstd = (midwp.wind-midwp.WS).std()
            print("Wind speed larger than {} m/s account for {:.4f}%".format(i, perc*100))
            print("Wind speed larger than {} m/s STD: {}".format(i, midstd))


###############################################################################
if __name__ == '__main__':
# =============================================================================
    filename = r".\result\test_final_wind.pkl"
    wind_evaluate(filename)
