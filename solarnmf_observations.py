#solarnmf_observations.py

#Will Barnes
#2 April 2015

#Import needed modules
import numpy as np
from scipy.ndimage.interpolation import rotate

class MakeData(object):
    """Class for constructing data, either simulation or observational, for use in NMF or BSS methods
    """
    
    def __init__(self,toption,input_type,**kwargs):
        """Constructor for MakeData class"""
        
        #Set observation matrix type
        self.toption = toption
        self.input_type = input_type
        
        #Check timeseries options
        if self.input_type == 'timeseries':
            if 'angle' not in kwargs:
                self.angle = 0
                print "angle not specified. Setting angle to ",self.angle
            else:
                self.angle = kwargs['angle']
                
        #Check simulation and data options
        if self.toption == 'simulation':
            if 'nx' not in kwargs:
                self.nx = 50
                print "nx not specified. Setting nx to ",self.nx
            else:
                self.nx = kwargs['nx']
            
            if 'ny' not in kwargs:
                self.ny = 50
                print "ny not specified. Setting ny to ",self.ny        
            else:
                self.ny = kwargs['ny']
            
            if 'p' not in kwargs:
                self.p = 5
                print "p not specified. Setting p to ",self.p
            else:
                self.p = kwargs['p']
        elif self.toption == 'data':
            if 'filename' not in kwargs:
                raise ValueError("'filename' needs to be specified in kwargs in order to load data.")
            else:
                self.filename = kwargs['filename']
        else:
            raise ValueError("Unknown observation option. Use either 'simulation' or 'data'.")
            
        
    def make_t_matrix(self):
        """Construct matrix or timeseries to factorize"""
        
        if self.toption == 'simulation':
            #Call the make gaussians function
            target,T = self.make_gaussians()
            
            if self.input_type == 'matrix':
                return target,T
                
            else:
                Tmat = self.ts_to_mat(T)
                return target,T,Tmat
                
        else:
            
            T = np.loadtxt(self.filename)
            T /= np.max(T)
            
            if self.input_type == 'matrix':
                return T
                
            elif self.input_type == 'timeseries':
                Tmat = self.ts_to_mat(T)
                return T,Tmat
                
            else:
                raise ValueError("Unknown input type. Use either 'timeseries' or 'matrix'.")
                
            
    def make_gaussians(self):
        """Construct simulated gaussian matrices or timeseries"""
            
        sigma = 0.08
        target = []
        
        if self.input_type == 'timeseries':
            #Initialize total
            T = 0.0
            #Simulate 1D timeseries
            #Create series of Gaussians
            t = np.linspace(-1,1,self.nx)
            for i in range(self.p):
                #Get random sign
                if np.random.rand() > 0.5:
                    mu_sign = -1.0
                else:
                    mu_sign = 1.0
                #Create the Gaussian
                temp = np.random.rand()*np.exp(-(t+mu_sign*np.random.rand())**2/(2.0*sigma**2))
                #Add to total
                T = T + temp
                #Save the components
                target.append(temp)
                
            #Normalize the T matrix
            T = T/np.max(T)
            
        elif self.input_type == 'matrix':
            #Calculate standard deviations in x and y
            sigma_x = sigma*np.ones((1,self.p))
            sigma_y = sigma*np.ones((1,self.p))

            #Generate random center positions for pulses (normalized to [0,1])
            centers = np.random.rand(self.p,2)

            #Preallocate for T matrix
            T = np.zeros((self.ny,self.nx))

            #Set up the X and Y meshes
            X,Y = np.meshgrid(np.linspace(0,1,self.nx),np.linspace(0,1,self.ny))

            #Calculate the Gaussian pulses and add them to the T matrix
            for i in range(self.p):
                #Calculate Gaussian pulse
                z = np.exp(-1/(2*(sigma_x[0,i])**2)*((X - centers[i,0])**2) - 1/(2*(sigma_y[0,i])**2)*((Y - centers[i,1])**2))
                #Add to total T matrix
                T = T + z
                #Save the events that make up the T matrix
                target.append(z)

            #Normalize the T matrix to one
            T = T/np.max(T)
            
        else:
            raise ValueError("Unknown input type. Use either 'timeseries' or 'matrix'.")
            
        return target,T
        
        
    def ts_to_mat(self,x):
        """Format timeseries as matrix for NMF or BSS factorization."""
            
        x_smooth = self.ts_smooth(x,window_length=11,window='hanning')
        
        x_mat = self.ts_by_gaussian(x_smooth,len(x_smooth),0.15)
        
        if angle != 0:
            x_mat  = self.crop_and_rotate(x_mat,self.angle)
            
        return x_mat
        
            
    def ts_smooth(x,**kwargs):
        """Smoothing algorithm from Scipy.org Cookbook. URL: http://wiki.scipy.org/Cookbook/SignalSmooth

        Parameters
        ----------
        x: 1darray
            one-dimensional time series that will be smmothed

        Keyword Parameters
        ------------------
        window_length: int
            integer that defines the window length for smoothing. Should be odd
        window: str
            type of window to be used. Choose from 'flat','hanning','hamming','bartlett',or 'blackman'

        Returns
        -------
        y: 1darray
            smoothed one-dimensional time series

        See also
        --------
        numpy.hanning,numpy.hamming,numpy.bartlett,numpy.blackman,numpy.convolve

        """
        
        #Provide some checks on our inputs to make sure they are valid

        if x.ndim != 1:
            raise ValueError("smooth_1d_window only accepts 1d timeseries data")

        if x.size < kwargs['window_length']:
            raise ValueError("Input vector needs to be bigger than window size")

        if not kwargs['window'] in ['flat','hanning','hamming','bartlett','blackman']:
            raise ValueError("Invalid window type. See documentation.")

        #Create s vector
        s = np.r_[x[kwargs['window_length']-1:0:-1],x,x[-1:-kwargs['window_length']:-1]]

        #Implement window type
        if kwargs['window'] == 'flat':
            w = np.ones(kwargs['window_length'],'d')
        else:
            w = eval('np.'+kwargs['window']+'('+str(kwargs['window_length'])+')')

        #Return the smoothed time series
        return np.convolve(w/w.sum(),s,mode='valid')
        
    
    def ts_by_gaussian(x,dim2,sigma):
        """Convert time series to matrix with some spread defined by a gaussian with standard deviation sigma."""
    
        #Set up Gaussian to filter results through
        t = np.linspace(0,1,dim2)
        xfilt = np.exp(-(t - 0.5)**2/(2*sigma**2))

        #Mimic matrix multiplication
        xfilt_mat = np.zeros([dim2,1])
        x_mat = np.zeros([len(x),1])
        xfilt_mat[:,0] = xfilt
        x_mat[:,0] = x

        #Return the filtered matrix
        return np.transpose(np.dot(x_mat,np.transpose(xfilt_mat)))
        
        
    def crop_and_rotate(x_mat,angle):
    
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
        
        