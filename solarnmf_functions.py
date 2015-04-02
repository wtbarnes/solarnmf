#solarnmf_functions.py

#Will Barnes
#27 January 2015

#Description:

#Inputs:

#Outputs:

#Import necessary modules
import numpy as np
import solarnmf_process_data as spd

def make_t_matrix(toption,**kwargs):
    """Set up the observation matrix either using a series of simulated gaussians or a dataset specified by a keyword argument.
    
    Parameters
    ----------
    toption: str
        Option that says whether this is a simulated ('simulation') observation (a series of gaussians) or an observation to be pulled from a file ('data')
    **kwargs
        nx: int
            number of columns for the simulated data set
        ny: int
            number of rows for the simulated data set
        p: int
            number of sources in the simulated data set
        
    Returns
    -------
    T: numpy two-dimensional array
        Matrix that represents the observation we want to factorize
    
    """
    #Check which option to use
    if "simulation" == toption:
        
        #Parse arguments for simulation
        for key in kwargs:
            if key == "nx":
                nx = int(kwargs[key])
            elif key == "ny":
                ny = int(kwargs[key])
            elif key == "p":
                p = int(kwargs[key])
            
        #Call function to make simulated results
        sim_result = make_gaussians(nx,ny,p,format=kwargs['format'])
        
        #If we are using Gauusian matrices, return the matrices
        if kwargs['format'] == 'matrix':
            #Return values 
            return {'target':sim_result['target'],'T':sim_result['T']}
        else:
            #Format vector as matrix
            x = sim_result['x']
            
            #Make sure data is normalized
            x = x/np.max(x)
        
            #Apply the smoothing function
            x_smooth = spd.smooth_1d_window(x,window_length=11,window='hanning')
        
            #Make a matrix representation of the time series
            x_mat = spd.ts2mat(x_smooth,len(x_smooth),0.1)
    
            #Rotate the important data along the diagonal
            x_mat_rot = spd.crop_and_rotate(x_mat,kwargs['angle'])
        
            #Return the rotated matrix and the original vector
            return {'T':x_mat_rot,'x':x,'target':sim_result['target']}
        
    if "data" == toption:
        for key in kwargs:
            if key == "filename":
                filename = kwargs[key]
            elif key == "file_format":
                file_format = kwargs[key]
        
        #Load the specified file
        x = np.loadtxt(filename)
        
        #Make sure data is normalized
        x = x/np.max(x)
        
        #Apply the smoothing function
        x_smooth = spd.smooth_1d_window(x,window_length=11,window='hanning')
        
        #Make a matrix representation of the time series
        x_mat = spd.ts2mat(x_smooth,len(x_smooth),0.1)
    
        #Rotate the important data along the diagonal
        x_mat_rot = spd.crop_and_rotate(x_mat,kwargs['angle'])
        
        #Return the rotated matrix and the original vector
        return {'T':x_mat_rot,'x':x}


def make_gaussians(nx,ny,p,**kwargs):
    
    #Check the format
    if kwargs['format'] == 'matrix':
        #Calculate standard deviations in x and y
        sigma_x = 0.15*np.ones((1,p))
        sigma_y = 0.15*np.ones((1,p))
    
        #Generate random center positions for pulses (normalized to [0,1])
        centers = np.random.rand(p,2)
    
        #Preallocate for composite and individual pulses
        T = np.zeros((ny,nx))
        target = np.zeros((ny,nx,p))
    
        #Set up the X and Y meshes
        X,Y = np.meshgrid(np.linspace(0,1,nx),np.linspace(0,1,ny))
    
        #Calculate the Gaussian pulses and add them to the T matrix
        for i in range(p):
            #Calculate Gaussian pulse
            z = np.exp(-1/(2*(sigma_x[0,i])**2)*((X - centers[i,0])**2) - 1/(2*(sigma_y[0,i])**2)*((Y - centers[i,1])**2))
            #Add to total T matrix
            T = T + z
            #Save the events that make up the T matrix
            target[:,:,i] = z
        
        #Normalize the T matrix to one
        T = T/np.max(T)
    
        #Return the X,Y,T and target matrices
        return {'target':target, 'T':T}
    else:
        #Create list for simulated gaussians
        target = []
        #Initialize total
        x = 0.0
        #Simulate 1D timeseries
        #Create series of Gaussians
        t = np.linspace(-1,1,nx)
        for i in range(p):
            #Get random sign
            if np.random.rand() > 0.5:
                mu_sign = -1.0
            else:
                mu_sign = 1.0
            #Create the Gaussian
            temp = np.random.rand()*np.exp(-(t+mu_sign*np.random.rand())**2/(2.0*0.1**2))
            #Add to total
            x = x + temp
            #Save the components
            target.append(temp)
            
        #Return the timeseries
        return {'target':target,'x':x}
    

def initialize_uva(nx,ny,q,r,r_iter,T):
    
    #Initialize U and V matrices
    utemp = np.random.rand(ny,q)
    vtemp = np.random.rand(q,nx)
    atemp = np.dot(utemp,vtemp)
    
    #Set artificial value for div
    div_final = 1000
    #Set convergence limit for div_limit
    div_limit = 1.0e-6
    
    #Begin loop to check divergence criteria
    for i in range(r):
        #Print some output
        print "Starting initialization iteration ",i
        #Call the minimizer
        utemp,vtemp,atemp,div = minimize_div(utemp,vtemp,T,atemp,r_iter,div_limit)
        #Get the last value of the divergence measure
        d_temp = div_temp[-1]
        
        #Check the new value of div
        if d_temp < div_final:
            div_final = d_temp
            u = utemp
            v = vtemp
        
        #Generate next random u and v
        utemp = np.random.rand(ny,q)
        vtemp = np.random.rand(q,nx)
        atemp = np.dot(utemp,vtemp)
        
    #Create the initial a matrix from the initial u and v matrices
    A = np.dot(u,v)
        
    #Return u and v values with lowest final div value
    return u,v,A
        
        
    
def minimize_div(u,v,T,A,max_i,div_limit):
    
    #Set epsilon parameter to make sure everything is non-negative
    eps = 1.0e-6
    
    #Initialize vector for divergence metric
    div = np.zeros(max_i)
    
    #Initialize counter and change in divergence
    i = 0
    delta_div = div_limit + 1
    
    #Normalize the columns of u
    u = normalize_ucols(u)
    
    #Calculate error matrix
    error = calc_error(T,u,v)
    
    #Begin loop to minimize divergence metric
    while i < max_i and delta_div > div_limit:
        
        #Update the U,V, and A matrices
        u,v,A = update_uva(u,v,T,A,error,eps)
        
        #Calculate divergence metric
        d = calculate_div(T,A)
        #Save the value
        div[i] = d
        #Calculate the change in the metric
        if i > 0:
            delta_div = np.fabs(d-div[i-1])
            
        #Print progress (skip for initialization runs)
        if max_i > 20:
            print "i = ",i
            print "div = ",d
        
        #Increment the counter 
        i = i+1
    
    #Truncate divergence vector
    div = div[0:i]
        
    #Save the results
    return u,v,A,div
    
    
def update_uva(u,v,T,A,error,eps):
    
    #Use Regularized HALS algorithm (see Ch. 4 of Cichoki et al.(2009))
    
    r,c = u.shape
    
    for i in range(c):
        #Get column of u and row of v
        uj = u[:,i]
        vj = v[i,:]
        #Calculate associate component of data
        yj = error + np.dot(uj,vj)
        #update v
        vj = np.dot(np.transpose(yj),uj)
        vj[np.where(vj<0.0)] = eps
        v[i,:] = vj
        #update u
        uj = np.dot(yj,vj)
        uj[np.where(uj<0.0)] = eps
        uj = uj/np.linalg.norm(uj)
        u[:,i] = uj
        #Recalculate error
        error = yj - np.outer(uj,vj)
        
    #Calculate A matrix
    A = np.dot(u,v)
    
    #Return the matrices
    return u,v,A
    

def update_u(u,v,T,A):
    #Lee & Seung (2000) update rules (for KL divergence)
    m,n = A.shape
    u = u*(np.dot(T/A,np.transpose(v)))/np.dot(np.ones((m,n)),np.transpose(v))
    
    #Return the u matrix
    return u
    
    
def update_v(u,v,T,A):
    #Lee & Seung (2000) update rules (for KL divergence)
    m,n = A.shape
    v = v*(np.dot(np.transpose(u),T/A))/np.dot(np.transpose(u),np.ones((m,n)))
    
    #Return the v matrix
    return v
    
    
def calculate_div(T,A):
    #Calculate the selected divergence metric
    #Kullback-Leibler divergence
    
    #Initialize the sum
    div = 0
    
    #Get dimensions of A
    m,n = A.shape
    
    #Loop to sum
    for i in range(m):
        for j in range(n):
            
            #Break up into terms
            term_1 = T[i,j]*np.log(T[i,j]/A[i,j])
            if T[i,j] == 0:
                term_1 = 0
            
            term_2 = -T[i,j]
            term_3 = A[i,j]
            
            #Add the terms
            div = div + term_1 + term_2 + term_3
            
            
    #Return the value
    return div
    
    
def normalize_ucols(u):
    #Normalize the columns of u
    r,c = u.shape
    
    for i in range(c):
        uc = u[:,i]
        uc = uc/np.linalg.norm(uc)
        u[:,i] = uc
        
    return u
    
def calc_error(T,u,v):
    return T - np.dot(u,v)    
