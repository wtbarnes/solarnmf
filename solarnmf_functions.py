#solarnmf_functions.py

#Will Barnes
#27 January 2015

#Description:

#Inputs:

#Outputs:

#Import necessary modules
import numpy as np

def make_t_matrix(toption,**kwargs):
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
        sim_result = make_gaussians(nx,ny,p)
        
        #Return values 
        return {'X':sim_result['X'],'Y':sim_result['Y'],'target':sim_result['target'],'T':sim_result['T']}
        
    if "data" == toption:
        for key in kwargs:
            if key == "filename":
                filename = kwargs[key]
            elif key == "file_format":
                file_format = kwargs[key]
        
        #DEBUG
        print "Input filename is ",filename
        print "Filename option not yet implemented"
    
    


def make_gaussians(nx,ny,p):
    
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
    return {'X':X, 'Y':Y, 'target':target, 'T':T}
    

def initialize_uv(nx,ny,q,r,r_iter,T):
    
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
        #Call the minimizer
        temp = minimize_div(utemp,vtemp,T,atemp,r_iter,div_limit)
        #Get the last value of the divergence measure
        div_temp = temp['div']
        d_temp = div_temp[-1]
        
        #Check the new value of div
        if d_temp < div_final:
            div_final = d_temp
            u = temp['u']
            v = temp['v']
        
        #Generate next random u and v
        utemp = np.random.rand(ny,q)
        vtemp = np.random.rand(q,nx)
        atemp = np.dot(utemp,vtemp)
        
    #Return u and v values with lowest final div value
    return {'u':u, 'v':v}
        
        
    
def minimize_div(u,v,T,A,max_i,div_limit):
    
    #Initialize vector for divergence metric
    div = np.zeros(max_i)
    
    #Initialize counter and change in divergence
    i = 0
    delta_div = div_limit + 1
    
    #Begin loop to minimize divergence metric
    while i < max_i and delta_div > div_limit:
        
        #Update the U,V, and A matrices
        updates = update_uva(u,v,T,A,i)
        u,v,A = updates['u'],updates['v'],updates['A']
        
        #Calculate divergence metric
        d = calculate_div(T,A)
        #Save the value
        div[i] = d
        #Calculate the change in the metric
        if i > 0:
            delta_div = np.fabs(d-div[i-1])
            
        #Print progress
        print "i = ",i
        print "div = ",d
        
        #Increment the counter 
        i = i+1
    
    #Truncate divergence vector
    div = div[0:i]
        
    #Save the results
    return {'u':u, 'v':v, 'A':A, 'div':div}
    
    
def update_uva(u,v,T,A,i):
    
    #Impose order of update rules (following JDB)
    #Alternating u,v, and A with alternating order
    if i % 2 == 0:
        u = update_u(u,v,T,A)
        A = np.dot(u,v)
        v = update_v(u,v,T,A)
        A = np.dot(u,v)
    else:
        v = update_v(u,v,T,A)
        A = np.dot(u,v)
        u = update_u(u,v,T,A)
        A = np.dot(u,v)
        
    #Return the matrices
    return {'u':u,'v':v,'A':A}
    

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
    
    