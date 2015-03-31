#solarnmf_plot_routines.py

#Will Barnes
#31 March 2015

#Import necessary modules
import numpy as np
import matplotlib.pyplot as plt

def plot_obsVpred(T,A):
    #Plot the total results for the observation and the prediction
    fig, axes = plt.subplots(1,2)
    axes[0].matshow(T,cmap='hot')
    axes[0].set_title(r'$T$ matrix (observation)',fontsize=18)
    axes[1].matshow(A,cmap='hot')
    axes[1].set_title(r'$A$ matrix (prediction)',fontsize=18)
    
    #Check if output filename is specified
    if 'print_fig_filename' in kwargs:
        plt.savefig(kwargs['print_fig_filename'],format='eps',dpi=1000)
    else:
        plt.show()


def plot_sim_targVpred(P,Q,u,v,target,**kwargs):
    #Make the reconstructed events
    #Set the number of columns for the plots
    if P > Q:
        num_cols = P
    else:
        num_cols = Q
    #Set up the figure and the axes
    fig, axes = plt.subplots(2,num_cols)
    #Loop over predictions and plot
    for i in range(Q):
        temp = np.outer(u[:,i],v[i,:])
        axes[1,i].matshow(temp,cmap='hot')
    
    #Loop over the targets that made up the initial 'observation' and plot
    for i in range(P):
        axes[0,i].matshow(target[:,:,i],cmap='hot')
        
    #Check if output filename is specified
    if 'print_fig_filename' in kwargs:
        plt.savefig(kwargs['print_fig_filename'],format='eps',dpi=1000)
    else:
        plt.show()

def plot_convergence(div,**kwargs):
    #Set up figure
    fig = plt.figure()
    #Set up axis
    ax = fig.gca()
    #Plot the divergence (versus iteration)
    ax.plot(div)
    #Set some labels and titles
    ax.set_title(r'Divergence metric',fontsize=18)
    ax.set_ylabel(r'$d(T,A)$',fontsize=18)
    ax.set_xlabel(r'iteration',fontsize=18)
    ax.set_yscale('log')
    
    #Check if output filename is specified
    if 'print_fig_filename' in kwargs:
        plt.savefig(kwargs['print_fig_filename'],format='eps',dpi=1000)
    else:
        plt.show()
 