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
results = snf.make_t_matrix("simulation",nx=NX,ny=NY,p=P,format='matrix')

#Initialize the U and V matrices
uva_initial = snf.initialize_uva(NX,NY,Q,10,10,results['T'])

#Start the minimizer
u,v,A,div = snf.minimize_div(uva_initial['u'],uva_initial['v'],results['T'],uva_initial['A'],500,1.0e-5)

#Plot the total results for the observation and the prediction
<<<<<<< HEAD
spr.plot_obsVpred(results['T'],A)

#Plot the reconstructed events
spr.plot_sim_targVpred(P,Q,u,v,results['target'])
    
#Plot the convergence
spr.plot_convergence(div)
=======
spr.plot_mat_obsVpred(results['T'],min_results['A'])

#Plot the reconstructed events
spr.plot_mat_targVpred(P,Q,min_results['u'],min_results['v'],results['target'])
    
#Plot the convergence
spr.plot_convergence(min_results['div'])
>>>>>>> 7059412e08f55fb35e4fd622290fa97c1a9eaad6
