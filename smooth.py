import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from utilities import df_combine
from cvxopt import matrix
from cvxopt.solvers import qp

# =============================================================================
# Carry the smooth procedure on discrete empirical GMF with specific functions
# =============================================================================


###############################################################################
class PGMF_generator(object):
    """ This class aim at smooth the empirical GMF function """
    def __init__(self, gmf, obs):
        self.gmf = gmf    # empirical gmf
        self.var = obs    # DDM observables 'nbrcs' or 'les'
        self.node = (15., 10.)  # abnormal transition point for 'nbrcs' or 'les'
        self.gmf_fit = {}       # save parametric gmf
        # put two functions here, is for automate smoothing and saving data
#        self.smooth_gmf()  # excute smooth_gmf function
#        self.smooth_gmf_buoy()  # smooth the buoy gmf
#        self.saveData()    # automate save the parametric data

    def _error_func_lower(self, x, t, y):
        """error function for f(x)=a+b*x**(-1)+c*x**(-2) """
        return x[0]+x[1]*pow(t, -1)+x[2]*pow(t, -2)-y

    def approximate(self, transition, scatter_lower, scatter_upper):
        """
            adjustment of indirect observations with constrains, 
            parabola need pass the point=(node, f0) and at this point
            the derivative equivalent dev
        """
        # fix function f(x)=a+b*x**(-1)+c*x**(-2), least square approximate
        # the function
        res_lsq = least_squares(self._error_func_lower, np.ones(3),
                                loss='cauchy', f_scale=0.1,
                                args=(scatter_lower.wind.values,
                                      scatter_lower[self.var].values),
                                bounds=(0., [1.0e3, 1.0e3, 1.0e3]))
        if not res_lsq.success:
            return
        coef = res_lsq.x
        # compute the function value and derivative of transition point
        f0 = coef[0]+coef[1]*pow(transition, -1)+coef[2]*pow(transition, -2)
        dev = -coef[1]*pow(transition, -2)-2*coef[2]*pow(transition, -3)
        # equation of observations for parabola: f(x) = a+b*x+c*x**2
        mat_B = np.mat(np.ones((len(scatter_upper), 3)))
        vec_l = np.mat(np.zeros(len(scatter_upper)))
        for i in range(len(scatter_upper)):
            vec_l[0, i] = scatter_upper[self.var].iloc[i]
            mat_B[i, 1] = scatter_upper.wind.iloc[i]
            mat_B[i, 2] = pow(scatter_upper.wind.iloc[i], 2)
        # equation of constrains
        mat_C = np.mat(np.ones((2, 3)))
        mat_C[0] = [1., transition, transition**2]
        mat_C[1] = [0., 1., 2*transition]
        vec_Wx = np.mat([-f0, -dev]).T
        # adjustment of indirect
        Nbb = mat_B.T*mat_B
        W = mat_B.T*vec_l.T
        try:
            Nbb_I = Nbb.I
            Ncc = mat_C*Nbb_I*mat_C.T
            Ncc_I = Ncc.I
        except np.linalg.LinAlgError:
            return
        mat_temp = Nbb_I*mat_C.T*Ncc_I
        parabola_coef = (Nbb_I-mat_temp*mat_C*Nbb_I)*W-mat_temp*vec_Wx
        # return two parametric models coefficients
        return (coef, np.array([parabola_coef[0, 0], parabola_coef[1, 0],
                                parabola_coef[2, 0]]))

    def approximate_qpsolve(self, transition, scatter_lower, scatter_upper):
        # fix function f(x)=a+b*x**(-1)+c*x**(-2), least square approximate
        # the function
        res_lsq = least_squares(self._error_func_lower, np.ones(3),
                                loss='cauchy', f_scale=0.1,
                                args=(scatter_lower.wind.values,
                                      scatter_lower[self.var].values),
                                bounds=(0., [1.0e3, 1.0e3, 1.0e3]))
        if not res_lsq.success:
            return
        coef = res_lsq.x
        # compute the function value and derivative of transition point
        f0 = coef[0]+coef[1]*pow(transition, -1)+coef[2]*pow(transition, -2)
        dev = -coef[1]*pow(transition, -2)-2*coef[2]*pow(transition, -3)
        # equation of observations for parabola: f(x) = a+b*x+c*x**2
        mat_B = matrix(1., (len(scatter_upper), 3))
        # vec_l = matrix(0., (len(scatter_upper), 1))
        vec_l = np.zeros((len(scatter_upper), 1))
        for i in range(len(scatter_upper)):
            vec_l[i] = scatter_upper[self.var].iloc[i]
            mat_B[i, 1] = scatter_upper.wind.iloc[i]
            mat_B[i, 2] = pow(scatter_upper.wind.iloc[i], 2)
        # #equation of constrains
        vec_l = matrix(vec_l)
        A = matrix(1.0, (2, 3))
        A[0, :] = [1., transition, transition**2]
        A[1, :] = [0., 1., 2*transition]
        b = matrix([f0, dev])
        # adjustment of indirect
        P = 2.*mat_B.T*mat_B
        q = -2.*mat_B.T*vec_l
        G = matrix([0., 0., 1.]).T
        h = matrix([0.])
        parabola_coef = qp(P, q, G, h, A, b)['x']
        print(parabola_coef)
        # return two parametric models coefficients
        return (coef, np.array([parabola_coef[0], parabola_coef[1],
                                parabola_coef[2]]))

    def transition_list(self, scope, step):
        """ list potential transition point  """
        if self.var == 'nbrcs':
            transition_list = np.arange(self.node[0]-scope,
                                        self.node[0]+scope, step)
        elif self.var == 'les':
            transition_list = np.arange(self.node[1]-scope,
                                        self.node[1]+scope, step)
        return transition_list
    
    def smooth_gmf(self):
        """ 
            gmf is dict:
            {incidence_angle:{'wind': wind,'observables':observables}}
            based on the potential transition points of nbrcs and les at
            15m/s and 10 m/s, try to search trainstion point arround these
            two values, by calculating the final std to determine the ultimate
            transition point
        """
        search_scope = 0.5     # search scope for transition point
        search_step = 0.01     # search step for tansition point
        trans_list = self.transition_list(search_scope, search_step)
        for angle in self.gmf:
            df_scatter = pd.DataFrame(self.gmf[angle])
            df_scatter = df_scatter.dropna()
            df_scatter = df_scatter[df_scatter.wind < 30.]
            # list test transition point
            coef_l, coef_u, std, node = None, None, None, None
            for transition in trans_list:
                scatter_lower = (df_scatter[df_scatter.wind <= transition])
                scatter_upper = (df_scatter[df_scatter.wind > transition])
                # approximation
                # coef = self.approximate(transition, scatter_lower,
                #                         scatter_upper)
                # approximate_qpsolve
                coef = self.approximate_qpsolve(transition, scatter_lower,
                                                scatter_upper)
                if coef is None:
                    continue
                coefl, coefu = coef
                # compute residual STD
                xl = scatter_lower.wind.values
                yl = scatter_lower[self.var].values
                diff_l = yl-coefl[0]-coefl[1]*pow(xl, -1)-coefl[2]*pow(xl, -2)
                xu = scatter_upper.wind.values
                yu = scatter_upper[self.var].values
                diff_u = yu-coefu[0]-coefu[1]*xu-coefu[2]*xu**2
                temp = np.append(diff_l, diff_u).std()
                # search least std and corresponding coefficent of functions
                if (std is None) | (std is None):
                    std = temp
                    coef_l, coef_u = coefl, coefu
                    node = transition
                    continue
                if std > temp:
                    std = temp
                    coef_l, coef_u = coefl, coefu
                    node = transition
            if node is None:
                continue
            # recording parametric gmf
            print('incidence={}, node={}, std={}'.format(angle, node, std))
            x_upper = np.arange(node, 35., 0.1)
            x_lower = np.arange(0.05, node, 0.1)
            
            y_upper = coef_u[0]+coef_u[1]*x_upper+coef_u[2]*x_upper**2
            y_lower = coef_l[0]+coef_l[1]*pow(x_lower, -1)+coef_l[2]*pow(x_lower, -2)
            x = np.append(x_lower, x_upper)
            y = np.append(y_lower, y_upper)
            self.gmf_fit[angle] = {'coef_l': coef_l,
                                   'clef_u': coef_u,
                                   'emp_'+self.var: df_scatter[self.var],
                                   'emp_wind': df_scatter.wind,
                                   'transition': node,
                                   'paras_wind': x,
                                   'paras_'+self.var: y,
                                   'std': std}

    def smooth_gmf_buoy(self):
        """ 
            gmf is dict:
            {incidence_angle:{'wind': wind,'observables':observables}}
            based on the potential transition points of nbrcs and les at
            15m/s and 10 m/s, try to search trainstion point arround these
            two values, by calculating the final std to determine the ultimate
            transition point
        """
        search_scope = 0.5     # search scope for transition point
        search_step = 0.01     # search step for tansition point
        trans_list = self.transition_list(search_scope, search_step)

        df_scatter = pd.DataFrame(self.gmf)
        df_scatter = df_scatter.dropna()
        df_scatter = df_scatter[df_scatter.wind < 30.]
        # list test transition point
        coef_l, coef_u, std, node = None, None, None, None
        for transition in trans_list:
            scatter_lower = df_scatter[df_scatter.wind <= transition]
            scatter_upper = df_scatter[df_scatter.wind > transition]
            # approximation
            # coef = self.approximate(transition, scatter_lower,
            #                         scatter_upper)
            # approximate_qpsolve
            coef = self.approximate_qpsolve(transition, scatter_lower,
                                            scatter_upper)
            if coef is None:
                continue
            coefl, coefu = coef
            # compute residual STD
            xl = scatter_lower.wind.values
            yl = scatter_lower[self.var].values
            diff_l = yl-coefl[0]-coefl[1]*pow(xl, -1)-coefl[2]*pow(xl, -2)
            xu = scatter_upper.wind.values
            yu = scatter_upper[self.var].values
            diff_u = yu-coefu[0]-coefu[1]*xu-coefu[2]*xu**2
            temp = np.append(diff_l, diff_u).std()
            # search least std and corresponding coefficent of functions
            if (std is None) | (std is None):
                std = temp
                coef_l, coef_u = coefl, coefu
                node = transition
                continue
            if std > temp:
                std = temp
                coef_l, coef_u = coefl, coefu
                node = transition
        if node is None:
            print("Smooth buoy empirical gmf failed!")
            return
        # recording parametric gmf
        print('node={}, std={}'.format(node, std))
        x_upper = np.arange(node, 35., 0.1)
        x_lower = np.arange(0.05, node, 0.1)
    
        y_upper = coef_u[0]+coef_u[1]*x_upper+coef_u[2]*x_upper**2
        y_lower = coef_l[0]+coef_l[1]*pow(x_lower, -1)+coef_l[2]*pow(x_lower, -2)
        x = np.append(x_lower, x_upper)
        y = np.append(y_lower, y_upper)
        self.gmf_fit = {'coef_l': coef_l,
                        'clef_u': coef_u,
                        'emp_'+self.var: df_scatter[self.var],
                        'emp_wind': df_scatter.wind,
                        'transition': node,
                        'paras_wind': x,
                        'paras_'+self.var: y,
                        'std': std}


###############################################################################
def save2file(var_tuple, filename):
    with open(filename, 'wb') as pfile:
        if isinstance(var_tuple, tuple):
            pickle.dump(var_tuple, pfile, pickle.HIGHEST_PROTOCOL)


###############################################################################
def paras_gmf(emp_gmf_file, save_filename):
    """ 
        This is main function in this module, aimming at generate
        parameterize empirical GMF function and ploting
        if plot_flag set true
    """
    # load empirical GMF and carry out parametric smoothing
    with open(emp_gmf_file, 'rb') as pfile:
        emp_gmf_nbrcs, emp_gmf_les = pickle.load(pfile)
    gmf_nbrcs = PGMF_generator(emp_gmf_nbrcs, 'nbrcs')
    gmf_nbrcs.smooth_gmf()  # excute smooth_gmf function
#    gmf_nbrcs.smooth_gmf_buoy()  # smooth the buoy gmf

    gmf_les = PGMF_generator(emp_gmf_les, 'les')
    gmf_les.smooth_gmf()  # excute smooth_gmf function
#    gmf_les.smooth_gmf_buoy()  # smooth the buoy gmf
    save2file((gmf_nbrcs.gmf_fit, gmf_les.gmf_fit), save_filename)


###############################################################################
def get_index(center, bin_width, var):
    """ get scatter of observables of CYGNSS data """
    # determine +/- one bin width sample
    lower_lim = center-bin_width
    upper_lim = center+bin_width
    ind = ((var >= lower_lim) & (var < upper_lim))
    # determin sample between 1 bin width and 2 bin width
    ex_lower_lim = lower_lim-bin_width
    ex_upper_lim = upper_lim+bin_width
    ex_ind = np.logical_or(((var >= ex_lower_lim) &
                           (var < lower_lim)),
                           ((var >= upper_lim) &
                           (var < ex_upper_lim)))
    return ind, ex_ind


###############################################################################
def plotting(incidence_angle, scatter_data, emp_gmf, para_gmf, flag,
             xlim, ylim, xtick, ytick, xlabel, ylabel, size=(4, 3)):
    """ plotting function """
    plt.figure(figsize=size)
    if flag:
        hb = plt.hexbin(scatter_data[0], scatter_data[1],
                        gridsize=(1000, 100), cmap='jet', bins='log')
    plt.plot(emp_gmf[0], emp_gmf[1], 'b-')
    plt.plot(para_gmf[0], para_gmf[1], 'k-')
    plt.axis([xlim[0], xlim[1], ylim[0], ylim[1]])
    cb = plt.colorbar(hb)
    cb.ax.set_ylabel('Density (%)')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xticks(range(xlim[0], xlim[1]+1, xtick))
    plt.yticks(range(ylim[0], ylim[1]+1, ytick))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if incidence_angle:
        plt.title('Incidence angle {} °'.format(incidence_angle))
    plt.tight_layout()
    plt.show()


###############################################################################
def scatter_density(data_file, filename, incident_list=[30.0]):
    """ plot the parametric GMF functions from files """
    if os.path.isfile(filename):
        with open(filename, 'rb') as pfile:
            gmf_nbrcs, gmf_les = pickle.load(pfile)
    # read scatter data for plotting
    if os.path.isfile(data_file):
        data = df_combine(pd.read_pickle(data_file),
                            flag='all', type='retrieval')
        # filter the observables
        data = data[data.ddm_nbrcs > 0.0]
        data = data[data.ddm_les > 0.0]
        data = data[data.range_corr_gain > 10.0]
    incd_bin_width = 2.
    for angle in incident_list:
        # extract the DDM observables at specific incidenct angle 
        ind, ex_ind = get_index(angle, incd_bin_width, data.incidence_angle)
        ind = ind | ex_ind
        # scatter plotting data
        sca_wind = data.loc[ind, 'WS'].values
        sca_nbrcs = data.loc[ind, 'ddm_nbrcs'].values
        sca_les = data.loc[ind, 'ddm_les'].values
        # empirical gmf data
        emp_nbrcs = gmf_nbrcs[angle]['emp_nbrcs']
        emp_nbrcs_wind = gmf_nbrcs[angle]['emp_wind']
        emp_les = gmf_les[angle]['emp_les']
        emp_les_wind = gmf_les[angle]['emp_wind']
        # parametric gmf data
        nbrcs = gmf_nbrcs[angle]['paras_nbrcs']
        wind_nbrcs = gmf_nbrcs[angle]['paras_wind']
        les = gmf_les[angle]['paras_les']
        wind_les = gmf_les[angle]['paras_wind']
        # plotting
        plotting(angle, (sca_nbrcs, sca_wind), (emp_nbrcs, emp_nbrcs_wind),
                 (nbrcs, wind_nbrcs), flag=True, xlim=(0, 500), ylim=(0, 25),
                 xtick=100, ytick=5, xlabel='DDMA', ylabel='wind speed (m/s)')
        plotting(angle, (sca_les, sca_wind), (emp_les, emp_les_wind),
                 (les, wind_les), flag=True, xlim=(0, 250), ylim=(0, 25),
                 xtick=50, ytick=5, xlabel='LES', ylabel='wind speed (m/s)')


###############################################################################
def plot_gmf(observable, filename, incident_list=[30.0], log_state=True):
    """
        plot the parametric GMF and empirical functions from files, once time
        just draw one type of observable's GMF curve, also can choose linear
        scale or log10 scale, control by log_state
    """
    # read parameric GMF from file
    if os.path.isfile(filename):
        with open(filename, 'rb') as pfile:
            gmf_fit = pickle.load(pfile)
    if observable == 'nbrcs':
        gmf_fit = gmf_fit[0]
    elif observable == 'les':
        gmf_fit = gmf_fit[1]
    # plot different parametrci GMF incidence angle from incident_list
    plt.figure(figsize=(3, 2.5))
    for angle in incident_list:
        # choose variables
        emp_obs = gmf_fit[angle]['emp_'+observable]
        emp_obs_wind = gmf_fit[angle]['emp_wind']
        obs = gmf_fit[angle]['paras_'+observable]
        wind_obs = gmf_fit[angle]['paras_wind']
        if log_state:
            ph = plt.plot(wind_obs, np.log10(obs), label='${}° $'.format(angle))
            continue
        ph = plt.plot(wind_obs, obs, label='${}° $'.format(angle))
        plt.plot(emp_obs_wind, emp_obs, '.', markersize=0.8,
                 color=ph[0].get_color(), label='_nolegend_')
    # decorate the figure
    if not log_state:
        if observable == 'nbrcs':
            ylim = (0, 200)
            ytick = 50
            label = "DDMA"
        elif observable == 'les':
            ylim = (0, 100)
            ytick = 20
            label = "LES"
        xtick = 5
        xlim = (0, 25)
        plt.ylabel(observable.upper()+' GMF')
    else:
        if observable == 'nbrcs':
            ylim = (0.5, 2)
            ytick = .5
            label = "DDMA"
        elif observable == 'les':
            ylim = (0, 2)
            ytick = .5
            label = "LES"
        xtick = 5
        xlim = (0, 30)
        plt.ylabel('log10({})'.format(label))
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xticks(np.arange(xlim[0], xlim[1]+1, xtick))
    plt.yticks(np.arange(ylim[0], ylim[1]+1, ytick))
    plt.xlabel('wind speed (m/s)')
    plt.legend(ncol=2, loc='best')
    plt.tight_layout()
    plt.show()
