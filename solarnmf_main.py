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

nx = 50
ny = 50
p = 3
q = 20
fn = '/data/datadrive2/AIA_fm_spectra/fp_frames/frames(AIA_131).2D_dir/frames(AIA_131).2D_frame700'

data = MakeData('data','matrix',filename=fn,nx=nx,ny=ny,p=p)
target,T = data.make_t_matrix()
minimizer = SeparateSources(T,q,'frobenius_norm_reg_sparse','ALS_reg_sparse')
u_temp,v_temp,A_temp = minimizer.initialize_uva()
u,v,A,div = minimizer.minimize_div(u_temp,v_temp,300)
plotter = MakeBSSPlots('data','matrix',u,v,A,T,div,target=target)
plotter.plot_obs_pred_total()
plotter.plot_obs_pred_sources()
plotter.plot_div()
