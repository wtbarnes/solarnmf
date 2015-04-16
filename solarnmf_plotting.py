#solarnmf_plotting.py

#Will Barnes
#3 April 2015

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage.interpolation import rotate

class MakeBSSPlots(object):

    def __init__(self,toption,input_type,u,v,A,T,div,**kwargs):
        self.toption = toption
        self.input_type = input_type
        self.u = u
        self.v = v
        self.A = A
        self.T = T
        self.div = div
        self.q = self.u.shape[1]
        
        self.fs = 18
        self.cm = 'hot'
        self.print_format = 'eps'
        self.print_dpi = 1000
        self.fig_size = (12,10)
        
        self.ny = kwargs['ny']
        self.nx = kwargs['nx']
        if 'angle' not in kwargs:
            self.angle = 0.0
        else:
            self.angle = kwargs['angle']
        
        self.A = self.rotate_back(A)
        self.get_components()
        self.ts_cut = np.unravel_index(self.A.argmax(),self.A.shape)[0]
        
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
                
    def rotate_back(self,mat):
        """Rotate back and reshape"""
        #rotate matrix back
        if self.angle != 0.0:
            mat_rot = rotate(mat,-self.angle)
        else:
            mat_rot = mat
        #get indices of rotated matrix
        ny_r,nx_r = mat_rot.shape
        #calculate differences
        delta_y = int(np.round((ny_r - self.ny)/2.0)) 
        delta_x = int(np.round((nx_r - self.nx)/2.0))
        #Return cut and rotated matrix
        return mat_rot[delta_y:((ny_r - delta_y)),delta_x:((nx_r - delta_x))]
    
    
    def get_components(self):
        """Separate A matrix into components"""
        self.components = []
        for i in range(self.q):
            self.components.append(self.rotate_back(np.outer(self.u[:,i],self.v[i,:])))
        
    
    def plot_obs_pred_total(self,**kwargs):
        """Plot original observation and recovered result"""
        if self.input_type == 'matrix':
            fig,ax = plt.subplots(1,2,figsize=self.fig_size)
            fig.tight_layout()
            imT = ax[0].imshow(self.T,cmap=self.cm)
            imA = ax[1].imshow(self.A,cmap=self.cm)
            ax[0].set_title(r'$T$, Observation',fontsize=self.fs)
            ax[1].set_title(r'$A$, Prediction',fontsize=self.fs)
            fig.colorbar(imT,cax=make_axes_locatable(ax[0]).append_axes("right","5%",pad="3%"),ticks=[np.min(self.T),(np.max(self.T)-np.min(self.T))/2.0,np.max(self.T)])
            fig.colorbar(imA,cax=make_axes_locatable(ax[1]).append_axes("right","5%",pad="3%"),ticks=[np.min(self.A),(np.max(self.A)-np.min(self.A))/2.0,np.max(self.A)])

        elif self.input_type == 'timeseries':
            fig = plt.figure(figsize=self.fig_size)
            ax = fig.gca()
            ax.plot(self.T,'.k',label='Observation')
            ax.plot(self.A[self.ts_cut,:],'r',label='Prediction')
            ax.set_title('Composite Time Series Comparison',fontsize=self.fs)
            ax.legend(loc=2)

        else:
            raise ValueError("Invalid input type option.")

        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename'],format=self.print_format,dpi=self.print_dpi)
        else:
            plt.show()


    def plot_obs_pred_sources(self,**kwargs):
        """Plot recovered sources against the original components"""
        try:
            rows = max(self.q,len(self.target))
            pairs = self.match_closest()
        except:
            rows = self.q
            pairs = []
            [pairs.append((i,i)) for i in range(self.q)]
            
            
        if self.input_type == 'matrix':
            fig,ax = plt.subplots(rows,2,figsize=self.fig_size)
            fig.tight_layout()
            ax[0,0].set_title(r'Sources',fontsize=self.fs)
            ax[0,1].set_title(r'Predictions',fontsize=self.fs)
            for i in range(rows):
                try:
                    im = ax[i,0].imshow(self.target[pairs[i][0]],cmap=self.cm)
                    ax[i,0].xaxis.set_ticklabels([])
                    ax[i,0].yaxis.set_ticklabels([])
                    fig.colorbar(im,cax=make_axes_locatable(ax[i,0]).append_axes("right","5%",pad="3%"),ticks=[np.min(self.target[i]),(np.max(self.target[i])-np.min(self.target[i]))/2.0,np.max(self.target[i])])
                except:
                    pass
                try:
                    im = ax[i,1].imshow(self.components[pairs[i][1]],cmap=self.cm)
                    ax[i,1].xaxis.set_ticklabels([])
                    ax[i,1].yaxis.set_ticklabels([])
                    fig.colorbar(im,cax=make_axes_locatable(ax[i,1]).append_axes("right","5%",pad="3%"),ticks=[np.min(self.components[i]),(np.max(self.components[i])-np.min(self.components[i]))/2.0,np.max(self.components[i])])
                except:
                    pass
                                
        elif self.input_type == 'timeseries':
            fig,ax = plt.subplots(rows,1,figsize=self.fig_size)
            fig.tight_layout()
            ax[0].set_title(r'Sources Reconstruction',fontsize=self.fs)
            for i in range(rows):
                ax[i].set_ylabel(r'$I$ (arb. units)',fontsize=self.fs)
                try:
                    ax[i].plot(self.target[pairs[i][0]],'.k',label='source')
                except:
                    pass
                try:
                    ax[i].plot(self.components[pairs[i][1]][self.ts_cut,:],'r',label='prediction')
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
        """Plot divergence metric as function of iteration"""
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.gca()
        ax.plot(self.div)
        ax.set_yscale('log')
        ax.set_xlim([0,len(self.div)])
        ax.set_title(r'Divergence Measure',fontsize=self.fs)
        ax.set_xlabel(r'iteration',fontsize=self.fs)
        ax.set_ylabel(r'$d(T,A)$',fontsize=self.fs)

        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename'],format=self.print_format,dpi=self.print_dpi)
        else:
            plt.show()
            
    def match_closest(self):
        """Create list of pairs of components and targets so that the plots correspond"""
        sources = range(self.q)
        pairs = []
        i_target = 0
        while sources != []:
            min_diff = 1.0e+50
            pairs.append(([],[]))
            
            for i in sources:
                if self.input_type == 'matrix':
                    diff = np.mean(np.fabs(self.target[i_target] - self.components[i]))
                else:
                    diff = np.mean(np.fabs(self.target[i_target] - self.components[i][self.ts_cut,:]))
                
                if diff < min_diff:
                    pairs.pop()
                    pairs.append((i_target,i)) 
                    min_diff = diff
                    
            sources.remove(pairs[i_target][1])
            i_target += 1
            
        return pairs
                
                