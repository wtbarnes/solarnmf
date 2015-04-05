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
q = 5

data = MakeData('simulation','matrix',nx=nx,ny=ny,q=q)
target,T = data.make_t_matrix()
minimizer = SeparateSources(T,q,'frobenius_norm','HALS')
u_temp,v_temp,A_temp = minimizer.initialize_uva()
u,v,A,div = minimizer.minimize_div(u_temp,v_temp,minimizer.max_i)
plotter = MakeBSSPlots('simulation','matrix',u,v,A,T,div,target=target)
plotter.plot_obs_pred_total()
plotter.plot_div()



