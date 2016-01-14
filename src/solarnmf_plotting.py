#solarnmf_plotting.py

#Will Barnes
#3 April 2015

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
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
        if 'q' not in kwargs:
            self.q = self.u.shape[1]
        else:
            self.q = kwargs['q']
        
        self.fs = 18
        self.cm = 'Blues'
        self.zero_tol = 1.e-5
        if 'print_format' not in kwargs:
            self.print_format = 'eps'
        else:
            self.print_format = kwargs['print_format']
        if 'dpi' not in kwargs:
            self.print_dpi = 1000
        else:
            self.print_dpi = kwargs['dpi']
        self.fig_size = (8,8)
        self.yaxis_format = FormatStrFormatter('%3.1f')
        
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
            plt.subplots_adjust(left=0.05,right=0.95,top=1.0,bottom=0.0,hspace=0.0,wspace=0.12)
            imT = ax[0].imshow(np.ma.masked_where(self.T<self.zero_tol*np.max(self.T),self.T),cmap=self.cm)
            imA = ax[1].imshow(np.ma.masked_where(self.A<self.zero_tol*np.max(self.A),self.A),cmap=self.cm)
            ax[0].set_title(r'$T$, Observation',fontsize=self.fs)
            ax[1].set_title(r'$A$, Prediction',fontsize=self.fs)
            ax[0].set_yticks([])
            ax[0].set_xticks([])
            ax[1].set_yticks([])
            ax[1].set_xticks([])
            cbar1 = fig.colorbar(imT,cax=make_axes_locatable(ax[0]).append_axes("right","5%",pad="3%"),ticks=[np.min(self.T),(np.max(self.T)-np.min(self.T))/2.0,np.max(self.T)],format=self.yaxis_format)
            cbar2 = fig.colorbar(imA,cax=make_axes_locatable(ax[1]).append_axes("right","5%",pad="3%"),ticks=[np.min(self.A),(np.max(self.A)-np.min(self.A))/2.0,np.max(self.A)],format=self.yaxis_format)

        elif self.input_type == 'timeseries':
            fig = plt.figure(figsize=self.fig_size)
            ax = fig.gca()
            plt.subplots_adjust(left=0.1,right=0.95,top=0.95,bottom=0.07,hspace=0.05)
            ax.plot(self.T,'.k',label='Observation')
            ax.plot(self.A[self.ts_cut,:],'r',label='Prediction')
            ax.set_xlabel(r'$t$ (au)',fontsize=self.fs)
            ax.set_ylabel(r'$I$ (au)',fontsize=self.fs)
            ax.set_title('Composite Comparison',fontsize=self.fs)
            ax.set_yticks([np.min(self.T),(np.max(self.T)-np.min(self.T))/2.0,np.max(self.T)])
            ax.yaxis.set_major_formatter(self.yaxis_format)
            ax.legend(loc=1)

        else:
            raise ValueError("Invalid input type option.")

        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename']+'.'+self.print_format,format=self.print_format,dpi=self.print_dpi)
            plt.close('all')
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
            fig,ax = plt.subplots(2,rows,figsize=self.fig_size)
            plt.subplots_adjust(left=0.05,right=0.98,top=1.0,bottom=0.0,hspace=0.0,wspace=0.1)
            ax[0,0].set_ylabel(r'Sources',fontsize=self.fs)
            ax[1,0].set_ylabel(r'Predictions',fontsize=self.fs)
            for i in range(rows):
                try:
                    tmp_mask = np.ma.masked_where(self.target[pairs[i][0]]<self.zero_tol*np.max(self.target[pairs[i][0]]),self.target[pairs[i][0]])
                    im = ax[0,i].imshow(tmp_mask,cmap=self.cm)
                    ax[0,i].set_yticks([])
                    ax[0,i].set_xticks([])
                    #fig.colorbar(im,cax=make_axes_locatable(ax[0,i]).append_axes("right","5%",pad="3%"),ticks=[np.min(self.target[i]),(np.max(self.target[i])-np.min(self.target[i]))/2.0,np.max(self.target[i])],format=self.yaxis_format)
                except:
                    pass
                try:
                    tmp_mask = np.ma.masked_where(self.target[pairs[i][1]]<self.zero_tol*np.max(self.target[pairs[i][1]]),self.target[pairs[i][1]])
                    im = ax[1,i].imshow(tmp_mask,cmap=self.cm)
                    ax[1,i].set_yticks([])
                    ax[1,i].set_xticks([])
                    #fig.colorbar(im,cax=make_axes_locatable(ax[1,i]).append_axes("right","5%",pad="3%"),ticks=[np.min(self.components[i]),(np.max(self.components[i])-np.min(self.components[i]))/2.0,np.max(self.components[i])],format=self.yaxis_format)
                except:
                    pass
                                
        elif self.input_type == 'timeseries':
            fig,ax = plt.subplots(rows,1,figsize=self.fig_size)
            plt.subplots_adjust(left=0.1,right=0.95,top=0.95,bottom=0.07,hspace=0.05)
            ax[0].set_title(r'Source Reconstruction',fontsize=self.fs)
            for i in range(rows):
                if i == int(rows/2.0):
                    ax[i].set_ylabel(r'$I$ (au)',fontsize=self.fs)
                try:
                    ax[i].plot(self.target[pairs[i][0]],'.k',label='source')
                except:
                    pass
                try:
                    ax[i].plot(self.components[pairs[i][1]][self.ts_cut,:],'r',label='prediction')
                    ax[i].set_yticks([0.0,(np.max(self.components[pairs[i][1]][self.ts_cut,:]) - np.min(self.components[pairs[i][1]][self.ts_cut,:]))/2.0,np.max(self.components[pairs[i][1]][self.ts_cut,:])])
                    ax[i].yaxis.set_major_formatter(self.yaxis_format)
                    ax[i].set_ylim([0,1])
                except:
                    pass
                if i == rows-1:
                    ax[i].set_xlabel(r'$t$ (au)',fontsize=self.fs)
                else:
                    ax[i].xaxis.set_ticklabels([])

            ax[0].legend(loc=1)

        else:
            raise ValueError("Invalid input type option")

        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename']+'.'+self.print_format,format=self.print_format,dpi=self.print_dpi)
            plt.close('all')
        else:
            plt.show()
            
            
    def plot_obs_pred_total_sources_ts(self,**kwargs):
        """Plot sources + total for observation and prediction"""
        if 'title' in kwargs:
            plot_title = kwargs['title']
        else:
            plot_title = 'Composite Comparison'
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.gca()
        plt.subplots_adjust(left=0.1,right=0.95,top=0.95,bottom=0.07,hspace=0.05)
        ax.plot(self.T,'.k',label='Observation')
        ax.plot(self.A[self.ts_cut,:],'r',label='Prediction')
        for i in range(self.q):
            ax.plot(self.components[i][self.ts_cut,:],'--b')
        ax.set_xlabel(r'$t$ (au)',fontsize=self.fs)
        ax.set_ylabel(r'$I$ (au)',fontsize=self.fs)
        ax.set_title(plot_title,fontsize=self.fs)
        ax.yaxis.set_major_formatter(self.yaxis_format)
        ax.set_ylim([0,1])
        ax.set_xlim([0,len(self.T)])
        ax.legend(loc=2)
        
        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename']+'.'+self.print_format,format=self.print_format,dpi=self.print_dpi)
            plt.close('all')
        else:
            plt.show()


    def plot_div(self,**kwargs):
        """Plot divergence metric as function of iteration"""
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.gca()
        ax.plot(self.div)
        ax.set_yscale('log')
        ax.set_xlim([0,len(self.div)])
        ax.set_ylim([np.min(self.div),np.max(self.div)])
        ax.set_title(r'Divergence Measure',fontsize=self.fs)
        ax.set_xlabel(r'iteration',fontsize=self.fs)
        ax.set_ylabel(r'$d(T,A)$',fontsize=self.fs)

        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename']+'.'+self.print_format,format=self.print_format,dpi=self.print_dpi)
            plt.close('all')
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
                
                
