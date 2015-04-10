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

nx = 250
ny = 1000
p = 3
q = p

data = MakeData('simulation','timeseries',nx=nx,ny=ny,p=p)
target,T = data.make_t_matrix()
minimizer = SeparateSources(T,q,'frobenius_norm_reg_sparse','ALS_reg_sparse')
u_temp,v_temp,A_temp = minimizer.initialize_uva()
u,v,A,div = minimizer.minimize_div(u_temp,v_temp,minimizer.max_i)
plotter = MakeBSSPlots('simulation','timeseries',u,v,A,T,div,target=target)
plotter.plot_obs_pred_total()
plotter.plot_obs_pred_sources()
plotter.plot_div()



