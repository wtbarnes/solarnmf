#solarnmf_plotting.py

#Will Barnes
#3 April 2015

import numpy as np
import matplotlib.pyplot as plt

class MakeBSSPlots(object):
    
    def __init__(self,toption,input_type,u,v,A,T,div,**kwargs):
        self.toption = toption
        self.input_type = input_type
        self.u = u
        self.v = v
        self.A = A
        self.T = T
        self.div = div
        self.ny,self.nx = self.T.shape
        dummy,self.q = self.u.shape
        self.fs = 18
        self.print_format = 'eps'
        self.print_dpi = 1000
        self.fig_size = (12,10)
        if self.toption == 'simulation':
            try:
                self.target = kwargs['target']
            except:
                raise ValueError("Please specify list of target sources when plotting simulation results.")
                
                
    def plot_obs_pred_total(self,**kwargs):
        if self.input_type == 'matrix':
            fig,ax = plt.subplots(1,2,figsize=self.fig_size)
            imT = ax[0].imshow(self.T,cmap='hot')
            imA = ax[1].imshow(self.A,cmap='hot')
            ax[0].set_title(r'$T$, Observation',fontsize=self.fs)
            ax[1].set_title(r'$A$, Prediction',fontsize=self.fs)
            fig.colorbar(imT,ax=ax[0])
            fig.colorbar(imA,ax=ax[1])
                
        elif self.input_type == 'timeseries':
            fig = plt.figure(figsize=self.fig_size)
            ax = fig.gca()
            ax.plot(self.T,'.k',label='Observation')
            ax.plot(self.A,'r',label='Prediction')
            ax.set_title('Composite Time Series Comparison',fontsize=self.fs)
            ax.legend(loc=2)
        
        else:
            raise ValueError("Invalid input type option.")
            
        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename'],format=self.print_format,dpi=self.print_dpi)
        else:
            plt.show()
            
            
    #def plot_obs_pred_sources(self,**kwargs):
        
    
            
    def plot_div(self,**kwargs):
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.gca()
        ax.plot(self.div)
        ax.set_yscale('log')
        ax.set_title(r'Divergence Measure',fontsize=self.fs)
        ax.set_xlabel(r'iteration',fontsize=self.fs)
        ax.set_ylabel(r'$d(T | A)$',fontsize=self.fs)
        
        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename'],format=self.print_format,dpi=self.print_dpi)
        else:
            plt.show()
                
        