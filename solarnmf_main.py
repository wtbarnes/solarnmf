#solarnmf_main.py

#Will Barnes
#27 January 2015

#Description:

#Inputs:

#Outputs:

###

import solarnmf_functions as snf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

NX = 100
NY = 100
P = 4
Q = 4

#Simulate some data
results = snf.make_t_matrix("simulation",nx=NX,ny=NY,p=P)

#Initialize the U and V matrices
uv_initial = snf.initialize_uv(NX,NY,Q,10,10,results['T'])

#Calculate the initial A matrix
a_initial = np.dot(uv_initial['u'],uv_initial['v'])

#Start the minimizer
min_results = snf.minimize_div(uv_initial['u'],uv_initial['v'],results['T'],a_initial,200,1.0e-5)

#DEBUG--plot the simulated gaussians as a test
fig1, axes1 = plt.subplots(1,2)
axes1[0].matshow(results['T'],cmap='hot')
axes1[0].set_title(r'$T$ matrix (observation)',fontsize=18)
axes1[1].matshow(min_results['A'],cmap='hot')
axes1[1].set_title(r'$A$ matrix (prediction)',fontsize=18)

#Make the reconstructed events
fig2, axes2 = plt.subplots(2,Q)
u = min_results['u']
v = min_results['v']
for i in range(Q):
    temp = np.outer(u[:,i],v[i,:])
    axes2[1,i].matshow(temp,cmap='hot')
    
target = results['target']    
for i in range(P):
    axes2[0,i].matshow(target[:,:,i],cmap='hot')
    
#Plot the convergence
fig3 = plt.figure()
ax3 = fig3.gca()
ax3.plot(min_results['div'])
ax3.set_title(r'Divergence metric',fontsize=18)
ax3.set_ylabel(r'$d(T,A)$',fontsize=18)
ax3.set_xlabel(r'iteration',fontsize=18)
    
plt.show()