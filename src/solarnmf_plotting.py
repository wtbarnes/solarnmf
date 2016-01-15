#solarnmf_plotting.py

#Will Barnes
#3 April 2015

import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.ndimage.interpolation import rotate

class MakeBSSPlots(object):

    def __init__(self,toption,input_type,u,v,A,T,div=None,q=None,angle=0.0,fig_size=(8,8),print_format='eps',print_dpi=1000,**kwargs):
        self.toption = toption
        self.input_type = input_type
        self.u = u
        self.v = v
        self.A = A
        self.T = T

        if div:
            self.div
        if not q:
            self.q = self.u.shape[1]
        else:
            self.q = q
        
        self.ny,self.nx = np.shape(T)
            
        #set optional member variables
        self.angle = angle
        self.print_format = print_format
        self.print_dpi = print_dpi
        self.fs = 18
        self.cm = 'Blues'
        self.zero_tol = 1.e-5
        self.fig_size = (8,8)
        self.yaxis_format = FormatStrFormatter('%3.1f')
        
        #Preprocessing
        self.A = self.rotate_back(A)
        self.get_components()
        self.ts_cut = np.unravel_index(self.A.argmax(),self.A.shape)[0]
        
        #Configure logger
        self.logger = logging.getLogger(type(self).__name__)
        
        #Check data type
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
            plt.tight_layout()
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
            plt.tight_layout()
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
            fig,ax = plt.subplots(2,rows,figsize=self.fig_size,sharex=True,sharey=True)
            plt.tight_layout()
            ax[0,0].set_ylabel(r'Sources',fontsize=self.fs)
            ax[1,0].set_ylabel(r'Predictions',fontsize=self.fs)
            for i in range(rows):
                try:
                    tmp_mask = np.ma.masked_where(self.target[pairs[i][0]]<self.zero_tol*np.max(self.target[pairs[i][0]]),self.target[pairs[i][0]])
                    im = ax[0,i].imshow(tmp_mask,cmap=self.cm)
                except IndexError:
                    self.logger.debug("Skipping source entry %d, out of range."%i)
                      
                try:
                    tmp_mask = np.ma.masked_where(self.target[pairs[i][1]]<self.zero_tol*np.max(self.target[pairs[i][1]]),self.target[pairs[i][1]])
                    im = ax[1,i].imshow(tmp_mask,cmap=self.cm)
                except IndexError:
                    self.logger.debug("Skipping source entry %d, out of range."%i)
                                
        elif self.input_type == 'timeseries':
            fig,ax = plt.subplots(rows,1,figsize=self.fig_size,sharex=True,sharey=True)
            plt.tight_layout()
            for i in range(rows):
                try:
                    ax[i].plot(self.target[pairs[i][0]],'.k',label='source')
                except IndexError:
                    self.logger.debug("Skipping source entry %d, out of range."%i)
                try:
                    ax[i].plot(self.components[pairs[i][1]][self.ts_cut,:],'r',label='prediction')
                    ax[i].set_yticks([0.0,(np.max(self.components[pairs[i][1]][self.ts_cut,:]) - np.min(self.components[pairs[i][1]][self.ts_cut,:]))/2.0,np.max(self.components[pairs[i][1]][self.ts_cut,:])])
                    ax[i].yaxis.set_major_formatter(self.yaxis_format)
                    ax[i].set_ylim([0,1])
                except IndexError:
                    self.logger.debug("Skipping source entry %d, out of range."%i)

            fig.text(0.07, 0.5, r'$I$ $\mathrm{(au)}$', ha='center',
                     va='center', rotation='vertical',fontsize=self.fs)
            ax[-1].set_xlabel(r'$t$ $\mathrm{(au)}$',fontsize=self.fs)
            ax[0].legend(loc='best')

        else:
            raise ValueError("Invalid input type option")

        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename']+'.'+self.print_format,format=self.print_format,dpi=self.print_dpi)
            plt.close('all')
        else:
            plt.show()
            
            
    def plot_obs_pred_total_sources_ts(self,**kwargs):
        """Plot sources + total for observation and prediction"""
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.gca()
        ax.plot(self.T,'.k',label='Observation')
        ax.plot(self.A[self.ts_cut,:],'r',label='Prediction')
        for i in range(self.q):
            ax.plot(self.components[i][self.ts_cut,:],'--b')
        ax.set_xlabel(r'$t$ (au)',fontsize=self.fs)
        ax.set_ylabel(r'$I$ (au)',fontsize=self.fs)
        ax.yaxis.set_major_formatter(self.yaxis_format)
        ax.set_ylim([0,1])
        ax.set_xlim([0,len(self.T)])
        ax.legend(loc='best')
        
        if 'print_fig_filename' in kwargs:
            plt.savefig(kwargs['print_fig_filename']+'.'+self.print_format,format=self.print_format,dpi=self.print_dpi)
            plt.close('all')
        else:
            plt.show()


    def plot_div(self,**kwargs):
        """Plot divergence metric as function of iteration"""
        try:
            fig = plt.figure(figsize=self.fig_size)
            ax = fig.gca()
            ax.plot(self.div)
            ax.set_yscale('log')
            ax.set_xlim([0,len(self.div)])
            ax.set_ylim([np.min(self.div),np.max(self.div)])
            ax.set_title(r'Divergence Measure',fontsize=self.fs)
            ax.set_xlabel(r'iteration',fontsize=self.fs)
            ax.set_ylabel(r'$d(T,A)$',fontsize=self.fs)
        except AttributeError:
            self.logger.error("Cannot plot divergence metric. self.div not set.")
            return

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
                
                
