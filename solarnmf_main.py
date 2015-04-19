#solarnmf_main.py

#Will Barnes
#27 January 2015

#Description:

#Inputs:

#Outputs:

###
import pickle
from solarnmf_observations import MakeData
from solarnmf_learn import SeparateSources
from solarnmf_plotting import MakeBSSPlots

nx = 100
ny = 100
p = 4
q = p
angle = 0.0
input_type = 'matrix'
data_option = 'simulation'
fn = '/home/wtb2/Desktop/ts_335_cut_test.txt'

params = {'eps':1.0e-4,'psi':1.0e-16,'sparse_u':0.125,'sparse_v':0.125,'reg_0':20.0,'reg_tau':50.0,'max_i':75,'r':5,'r_iter':10}
params['lambda_1'] = 0.0001
params['lambda_2'] = 0.0001
params['alpha'] = 0.8
params['l_toeplitz'] = 5
params['div_measure'] = 'multiplicative_reg_sparse'
params['update_rules'] = 'chen_cichocki_reg_sparse'

path2data = './data/'+input_type+'_'+data_option+'_q'+str(q)+'.uva' 

data = MakeData(data_option,input_type,file=fn,nx=nx,ny=ny,p=p,angle=angle)
target,T = data.make_t_matrix()

minimizer = SeparateSources(T,q,params,verbose=True)
u_temp,v_temp,A_temp = minimizer.initialize_uva()
u,v,A,div = minimizer.minimize_div(u_temp,v_temp,minimizer.max_i)
with open(path2data,'wb') as f:
    pickle.dump([u,v,A,T,target,div],f)
f.close()

ny_r,nx_r = T.shape[0],T.shape[0]
plotter = MakeBSSPlots(data_option,input_type,u,v,A,T,div,angle=angle,target=target,ny=ny_r,nx=nx_r)
plotter.plot_obs_pred_total()
plotter.plot_obs_pred_sources()
plotter.plot_div()
