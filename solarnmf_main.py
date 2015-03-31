#solarnmf_main.py

#Will Barnes
#27 January 2015

#Description:

#Inputs:

#Outputs:

###

import solarnmf_functions as snf
import solarnmf_plot_routines as spr
import numpy as np

NX = 100
NY = 100
P = 4
Q = 3

#Simulate some data
results = snf.make_t_matrix("simulation",nx=NX,ny=NY,p=P)

#Initialize the U and V matrices
uv_initial = snf.initialize_uv(NX,NY,Q,10,10,results['T'])

#Calculate the initial A matrix
a_initial = np.dot(uv_initial['u'],uv_initial['v'])

#Start the minimizer
min_results = snf.minimize_div(uv_initial['u'],uv_initial['v'],results['T'],a_initial,500,1.0e-5)

#Plot the total results for the observation and the prediction
spr.plot_obsVpred(results['T'],min_results['A'])

#Plot the reconstructed events
spr.plot_sim_targVpred(P,Q,min_results['u'],min_results['v'],results['target'])
    
#Plot the convergence
spr.plot_convergence(min_results['div'])