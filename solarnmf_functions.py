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
                
        #DEBUG
        print "(nx,ny,p) = (",nx,",",ny,",",p,")"
        
    if "data" == toption:
        if key == "filename":
            filename = kwargs[key]
        #DEBUG
        print "Input filename is ",filename
    
    


def make_gaussians(nx,ny,p):
    
    #Calculate standard deviations in x and y
    sigma_x = 0.15*np.ones(1,p)
    sigma_y = 0.15*np.ones(1,p)
    
    #Generate random center positions for pulses
    centers = np.random.rand(p,2)
    
    #Preallocate for composite and individual pulses
    T = np.zeros((ny,nx))
    #targ = 