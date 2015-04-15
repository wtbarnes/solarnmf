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
        self.ts_cut = np.unravel_index(self.A.argmax(),self.A.shape)[0]
        
        self.fs = 18
        self.cm = 'hot'
        self.print_format = 'eps'
        self.print_dpi = 1000
        self.fig_size = (12,10)
        
        if 'angle' not in kwargs:
            self.angle = 0.0
        else:
            self.angle = kwargs['angle']
        
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
                
        self.components = []
        self.make_components()


    def plot_obs_pred_total(self,**kwargs):
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
        try:
            rows = max(self.q,len(self.target))
        except:
            rows = self.q

        if self.input_type == 'matrix':
            fig,ax = plt.subplots(rows,2,figsize=self.fig_size)
            fig.tight_layout()
            ax[0,0].set_title(r'Sources',fontsize=self.fs)
            ax[0,1].set_title(r'Predictions',fontsize=self.fs)
            for i in range(rows):
                try:
                    im = ax[i,0].imshow(self.target[i],cmap=self.cm)
                    ax[i,0].xaxis.set_ticklabels([])
                    ax[i,0].yaxis.set_ticklabels([])
                    fig.colorbar(im,cax=make_axes_locatable(ax[i,0]).append_axes("right","5%",pad="3%"),ticks=[np.min(self.target[i]),(np.max(self.target[i])-np.min(self.target[i]))/2.0,np.max(self.target[i])])
                except:
                    pass
                try:
                    ai = np.outer(self.u[:,i],self.v[i,:])
                    im = ax[i,1].imshow(ai,cmap=self.cm)
                    ax[i,1].xaxis.set_ticklabels([])
                    ax[i,1].yaxis.set_ticklabels([])
                    fig.colorbar(im,cax=make_axes_locatable(ax[i,1]).append_axes("right","5%",pad="3%"),ticks=[np.min(ai),(np.max(ai)-np.min(ai))/2.0,np.max(ai)])
                except:
                    pass
                                
        elif self.input_type == 'timeseries':
            fig,ax = plt.subplots(rows,1,figsize=self.fig_size)
            fig.tight_layout()
            ax[0].set_title(r'Sources Reconstruction',fontsize=self.fs)
            for i in range(rows):
                ax[i].set_ylabel(r'$I$ (arb. units)',fontsize=self.fs)
                try:
                    ax[i].plot(self.target[i],'.k',label='source')
                except:
                    pass
                try:
                    ax[i].plot(self.components[i][self.ts_cut,:],'r',label='prediction')
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
            
    def make_components(self):
        """Rotate matrices back"""
        
        if self.angle != 0:
            self.A  = self.crop_and_rotate(self.A,self.angle)
            for i in range(self.q):
                self.components.append(self.crop_and_rotate(np.outer(self.u[:,i],self.v[i,:]),self.angle))
        else:
            for i in range(self.q):
                self.components.append(np.outer(self.u[:,i],self.v[i,:]))
            
    def crop_and_rotate(self,x_mat,angle):
    
        #Find the backgound value
        bg_val = np.min(x_mat[np.where(x_mat>np.max(x_mat)/100.0)])
    
        #Rotate the image and interpolate as necessary
        x_rot = rotate(x_mat,angle)
    
        #Find bounds by subtracting out background
        row_bounds,col_bounds = np.where(x_rot>bg_val)
        top = np.min(row_bounds)
        bottom = np.max(row_bounds)
        left = np.min(col_bounds)
        right = np.max(col_bounds)
    
        #Return trimmed matrix
        return x_rot[top:bottom,left:right]
