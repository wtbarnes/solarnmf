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
fn = '/data/datadrive2/AIA_fm_spectra/fp_frames/frames(AIA_131).2D_dir/frames(AIA_131).2D_frame700'

params = {'eps':1.0e-6,'psi':1.0e-16,'sparse_u':0.25,'sparse_v':0.25,'reg_0':20.0,'reg_tau':50.0,'max_i':1000,'r':10,'r_iter':10}
params['div_measure'] = 'frobenius_norm_reg_sparse'
params['update_rules'] = 'ALS_reg_sparse'

data = MakeData('simulation','matrix',filename=fn,nx=nx,ny=ny,p=p)
target,T = data.make_t_matrix()
minimizer = SeparateSources(T,q,params)
u_temp,v_temp,A_temp = minimizer.initialize_uva()
u,v,A,div = minimizer.minimize_div(u_temp,v_temp,minimizer.max_i)
plotter = MakeBSSPlots('simulation','matrix',u,v,A,T,div,target=target)
plotter.plot_obs_pred_total()
plotter.plot_obs_pred_sources()
plotter.plot_div()
