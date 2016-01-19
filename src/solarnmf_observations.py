#solarnmf_observations.py

#Will Barnes
#2 April 2015

#Import needed modules
import logging
import numpy as np
from scipy.ndimage.interpolation import rotate

class MakeData(object):
    """Class for constructing data, either simulation or observational, for use in NMF or BSS methods
    """

    def __init__(self,toption,input_type,nx=50,ny=50,p=5,angle=0.0,ngrid_y=10,ngrid_x=10,**kwargs):
        """Constructor for MakeData class"""

        #Set observation matrix type
        self.toption = toption
        self.input_type = input_type

        self.psi = 1.0e-16
        self.ngrid_y = ngrid_y
        self.ngrid_x = ngrid_x
        self.noise_level = 0.05
        
        #Configure logger
        self.logger = logging.getLogger(type(self).__name__)

        #Check timeseries options
        if self.input_type == 'timeseries':
            self.angle = angle
            self.logger.info("Setting angle = %f"%(self.angle))

        #Check simulation and data options
        if self.toption == 'simulation':
            self.nx = nx
            self.logger.info("Setting nx = %d"%(self.nx))
            self.ny = ny
            self.logger.info("Setting ny = %d"%(self.ny))
            self.p = p
            self.logger.info("Setting p = %d"%(self.p))

            self.grid_y = int(self.ny/self.ngrid_y)
            self.grid_x = int(self.nx/self.ngrid_x)
            self.sigma_y = float(self.grid_y/2.0)
            self.sigma_x = float(self.grid_x/2.0)

        elif self.toption == 'data':
            if 'file' not in kwargs:
                raise ValueError("'file' needs to be specified in kwargs by either a path or a variable in order to load data.")
            else:
                self.file = kwargs['file']
        else:
            raise ValueError("Unknown observation option. Use either 'simulation' or 'data'.")


    def make_t_matrix(self):
        """Construct matrix or timeseries to factorize"""

        if self.toption == 'simulation':
            #Call the make gaussians function
            target,T = self.make_gaussians()
            T[np.where(T<self.psi)] = self.psi

            if self.input_type == 'matrix':
                return target,T

            else:
                Tmat = self.ts_to_mat(T)
                Tmat[np.where(Tmat<self.psi)] = self.psi
                return target,T,Tmat

        else:

            if type(self.file) is str:
                T = np.loadtxt(self.file)
            else:
                T = self.file
            T /= np.max(T)
            T[np.where(T<self.psi)] = self.psi

            if self.input_type == 'matrix':
                return T

            elif self.input_type == 'timeseries':
                self.ny = T.shape[0]
                self.sigma_y = float(int(self.ny/self.ngrid_y)/2.0)
                Tmat = self.ts_to_mat(T)
                return T,Tmat

            else:
                raise ValueError("Unknown input type. Use either 'timeseries' or 'matrix'.")


    def make_gaussians(self):
        """Construct simulated gaussian matrices or timeseries"""

        target = []
        centers = []

        #Seed random number generator
        np.random.seed()

        if self.input_type == 'timeseries':
            #Initialize total
            T = np.zeros(self.nx)
            #Simulate 1D timeseries
            #Create series of Gaussians
            for i in range(self.ngrid_x):
                centers.append((i+0.5)*self.grid_x)

            t = np.linspace(0,self.nx-1,self.nx)
            n_centers = self.ngrid_x

            for i in range(self.p):
                i_center = np.random.randint(0,n_centers-1)
                #Create the Gaussian
                temp = np.exp(-(t-centers[i_center])**2/(2.0*self.sigma_x**2))
                #Add to total
                T = T + temp
                #Save the components
                target.append(temp)
                #Update centers list
                centers.remove(centers[i_center])
                n_centers += -1


        elif self.input_type == 'matrix':
            #Calculate standard deviations in x and y
            sigma_x = self.sigma_x*np.ones((1,self.p))
            sigma_y = self.sigma_y*np.ones((1,self.p))

            #Generate random center positions for pulses (normalized to [0,1])
            for i in range(self.ngrid_y):
                for j in range(self.ngrid_x):
                    centers.append(((i + 0.5)*self.grid_y,(j + 0.5)*self.grid_x))

            n_centers  = self.ngrid_y*self.ngrid_x

            #Preallocate for T matrix
            T = np.zeros((self.ny,self.nx))

            #Set up the X and Y meshes
            X,Y = np.meshgrid(np.linspace(0,self.nx-1,self.nx),np.linspace(0,self.ny-1,self.ny))

            #Calculate the Gaussian pulses and add them to the T matrix
            for i in range(self.p):
                i_center = np.random.randint(0,n_centers-1)
                #Calculate Gaussian pulse
                z = np.exp(-1/(2*(sigma_x[0,i])**2)*((X - centers[i_center][1])**2) - 1/(2*(sigma_y[0,i])**2)*((Y - centers[i_center][0])**2))
                #Add to total T matrix
                T = T + z
                #Save the events that make up the T matrix
                target.append(z)
                #Update centers list and length
                centers.remove(centers[i_center])
                n_centers += -1


        else:
            raise ValueError("Unknown input type. Use either 'timeseries' or 'matrix'.")

        T = self.add_noise(T)
        T = T/np.max(T)

        return target,T


    def ts_to_mat(self,x):
        """Format timeseries as matrix for NMF or BSS factorization."""

        #x_smooth = self.ts_smooth(x,window_length=11,window='hanning')

        x_mat = self.ts_by_gaussian(x)

        if self.angle != 0:
            x_mat  = rotate(x_mat,self.angle)

        return x_mat


    def ts_smooth(self,x,**kwargs):
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


    def ts_by_gaussian(self,x):
        """Convert time series to matrix with some spread defined by a gaussian with standard deviation sigma."""

        #Set up Gaussian to filter results through
        t = np.linspace(0,self.ny-1,self.ny)
        mu = float(self.ny-1.0)/2.0
        xfilt = np.exp(-(t - mu)**2/(2.0*self.sigma_y**2))

        #Mimic matrix multiplication
        xfilt_mat = np.zeros([self.ny,1])
        x_mat = np.zeros([len(x),1])
        xfilt_mat[:,0] = xfilt
        x_mat[:,0] = x

        #Return the filtered matrix
        return np.transpose(np.dot(x_mat,np.transpose(xfilt_mat)))


    def add_noise(self,mat):
        """Add noise to matrix"""

        np.random.seed()
        if self.input_type == 'matrix':
            nr,nc = mat.shape
            mat += self.noise_level*np.random.rand(nr,nc)
        else:
            n = mat.shape[0]
            mat += self.noise_level*np.random.rand(n)

        return mat
