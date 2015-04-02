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
u_initial,v_initial,A_initial = snf.initialize_uva(NX,NY,Q,10,50,results['T'])

#Start the minimizer
u,v,A,div = snf.minimize_div(u_initial,v_initial,results['T'],A_initial,500,1.0e-5)

#Plot the total results for the observation and the prediction
spr.plot_obsVpred(results['T'],A)

#Plot the reconstructed events
spr.plot_sim_targVpred(P,Q,u,v,results['target'])
    
#Plot the convergence
spr.plot_convergence(div)



