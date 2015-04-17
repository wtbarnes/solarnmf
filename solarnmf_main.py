#solarnmf_main.py

#Will Barnes
#27 January 2015

#Description:

#Inputs:

#Outputs:

###

from solarnmf_observations import MakeData
from solarnmf_learn import SeparateSources
from solarnmf_plotting import MakeBSSPlots

nx = 500
ny = 500
p = 15
q = p
angle = 45.0
fn = '/home/wtb2/Desktop/ts_335_cut_test.txt'

params = {'eps':1.0e-4,'psi':1.0e-16,'sparse_u':0.125,'sparse_v':0.125,'reg_0':20.0,'reg_tau':50.0,'max_i':100,'r':5,'r_iter':10}
params['lambda_1'] = 0.0001
params['lambda_2'] = 0.0001
params['alpha'] = 0.8
params['l_toeplitz'] = 5
params['div_measure'] = 'multiplicative_reg_sparse'
params['update_rules'] = 'chen_cichocki_reg_sparse'

data = MakeData('data','timeseries',file=fn,nx=nx,ny=ny,p=p,angle=angle)
T,Tmat = data.make_t_matrix()

minimizer = SeparateSources(Tmat,q,params)
u_temp,v_temp,A_temp = minimizer.initialize_uva()
u,v,A,div = minimizer.minimize_div(u_temp,v_temp,minimizer.max_i)

ny_r,nx_r = T.shape[0],T.shape[0]
plotter = MakeBSSPlots('data','timeseries',u,v,A,T,div,angle=angle,Tmat=Tmat,ny=ny_r,nx=nx_r)
plotter.plot_obs_pred_total()
plotter.plot_obs_pred_sources()
plotter.plot_div()
