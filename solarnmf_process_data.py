#solarnmf_process_data.py

#Will Barnes
#30 March 2014

#Import necessary modules
import numpy as np

def smooth_1d_fft(t,x,freq_thresh):
    
    #Perform FFT on timeseries data
    x_fft = np.fft.fft(x)
    
    #Create frequency vector from time vector
    freq = np.fft.fftfreq(len(t))
    
    #Filter out noisey, high frequency oscillations
    x_fft[np.where(np.fabs(freq) > freq_thresh)] = 0.0
    
    #Perform the inverse FFT
    x_ifft = np.fft.ifft(x_fft)
    
    #Return the real part of the FFT smoothed vector
    return x_ifft.real
    
    
def smooth_1d_window(x,**kwargs):
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
        raise ValueError, "smooth_1d_window only accepts 1d timeseries data"
        
    if x.size < kwargs['window_length']:
        raise ValueError, "Input vector needs to be bigger than window size"
        
    if not kwargs['window'] in ['flat','hanning','hamming','bartlett','blackman']:
        raise ValueError, "Invalid window type. See documentation."
        
    #Create s vector
    s = np.r_[x[kwargs['window_length']-1:0:-1],x,x[-1:-kwargs['window_length']:-1]]
    
    #Implement window type
    if kwargs['window'] == 'flat':
        w = np.ones(kwargs['window_length'],'d')
    else:
        w = eval('np.'+kwargs['window']+'('+str(kwargs['window_length'])+')')
    
    #Evaluate the smoothed time series
    y = np.convolve(w/w.sum(),s,mode='valid')
    
    #Return the smoothed vector
    return y
    
    