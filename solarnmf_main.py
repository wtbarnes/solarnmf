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

nx = 100
ny = 100
p = 5
q = p
fn = '/Users/willbarnes/Desktop/frames(AIA_131).2D_lt.v2m'

params = {'eps':1.0e-6,'psi':1.0e-16,'sparse_u':0.125,'sparse_v':0.125,'reg_0':20.0,'reg_tau':50.0,'max_i':1000,'r':10,'r_iter':10}
params['lambda_1'] = 0.001
params['lambda_2'] = 0.001
params['alpha'] = 0.8
params['l_toeplitz'] = 5
params['div_measure'] = 'multiplicative_reg_sparse'
params['update_rules'] = 'chen_cichocki_reg_sparse'

data = MakeData('simulation','matrix',filename=fn,nx=nx,ny=ny,p=p)
target,T = data.make_t_matrix()
minimizer = SeparateSources(T,q,params)
u_temp,v_temp,A_temp = minimizer.initialize_uva()
u,v,A,div = minimizer.minimize_div(u_temp,v_temp,minimizer.max_i)
plotter = MakeBSSPlots('simulation','matrix',u,v,A,T,div,target=target)
plotter.plot_obs_pred_total()
plotter.plot_obs_pred_sources()
plotter.plot_div()
