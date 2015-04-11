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
        dummy,self.q = self.u.shape
        self.fs = 18
        self.cm = 'hot'
        self.print_format = 'eps'
        self.print_dpi = 1000
        self.fig_size = (12,10)
        if self.toption == 'simulation':
            try:
                self.target = kwargs['target']
            except:
                raise ValueError("Please specify list of target sources when plotting simulation results.")
        if self.input_type == 'timeseries':
            try:
                self.ny = kwargs['Tmat'].shape[0]
            except:
                raise ValueError("Please specify matrix representation of time series when using 1D representation.")
                
                
    def plot_obs_pred_total(self,**kwargs):
        if self.input_type == 'matrix':
            fig,ax = plt.subplots(1,2,figsize=self.fig_size)
            imT = ax[0].imshow(self.T,cmap=self.cm)
            imA = ax[1].imshow(self.A,cmap=self.cm)
            ax[0].set_title(r'$T$, Observation',fontsize=self.fs)
            ax[1].set_title(r'$A$, Prediction',fontsize=self.fs)
            fig.colorbar(imT,ax=ax[0])
            fig.colorbar(imA,ax=ax[1])
                
        elif self.input_type == 'timeseries':
            fig = plt.figure(figsize=self.fig_size)
            ax = fig.gca()
            ax.plot(self.T,'.k',label='Observation')
            ax.plot(self.A[int(self.ny/2),:],'r',label='Prediction')
            ax.set_title('Composite Time Series Comparison',fontsize=self.fs)
            ax.legend(loc=2)
        
        else:
            raise ValueError("Invalid input type option.")
            
        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename'],format=self.print_format,dpi=self.print_dpi)
        else:
            plt.show()
            
            
    def plot_obs_pred_sources(self,**kwargs):
        rows = max(self.q,len(self.target))
        
        if self.input_type == 'matrix':
            fig,ax = plt.subplots(rows,2,figsize=self.fig_size)
            ax[0,0].set_title(r'Sources',fontsize=self.fs)
            ax[0,1].set_title(r'Predictions',fontsize=self.fs)
            for i in range(rows):
                try:
                    ax[i,0].imshow(self.target[i],cmap=self.cm)
                    ax[i,0].xaxis.set_ticklabels([])
                    ax[i,0].yaxis.set_ticklabels([])
                    fig.colorbar()
                except:
                    pass
                try:
                    ax[i,1].imshow(np.outer(self.u[:,i],self.v[i,:]),cmap=self.cm)
                    ax[i,1].xaxis.set_ticklabels([])
                    ax[i,1].yaxis.set_ticklabels([])
                    fig.colorbar()
                except:
                    pass
        elif self.input_type == 'timeseries':
            fig,ax = plt.subplots(rows,1,figsize=self.fig_size)
            ax[0].set_title(r'Sources Reconstruction',fontsize=self.fs)
            for i in range(rows):
                ax[i].set_ylabel(r'$I$ (arb. units)',fontsize=self.fs)
                try:
                    ax[i].plot(self.target[i],'.k',label='source')
                except:
                    pass
                try:
                    ax[i].plot(np.outer(self.u[:,i],self.v[i,:])[int(self.ny/2),:],'r',label='prediction')
                except:
                    pass
                if i == rows-1:
                    ax[i].set_xlabel(r'$t$ (arb. units)',fontsize=self.fs)
                
            ax[0].legend(loc=2)
            
        else:
            raise ValueError("Invalid input type option")
                
        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename'],format=self.print_format,dpi=self.print_dpi)
        else:
            plt.show()
    
            
    def plot_div(self,**kwargs):
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.gca()
        ax.plot(self.div)
        ax.set_yscale('log')
        ax.set_title(r'Divergence Measure',fontsize=self.fs)
        ax.set_xlabel(r'iteration',fontsize=self.fs)
        ax.set_ylabel(r'$d(T,A)$',fontsize=self.fs)
        
        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename'],format=self.print_format,dpi=self.print_dpi)
        else:
            plt.show()
                
        