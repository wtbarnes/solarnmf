#solarnmf_learn.py

#Will Barnes
#2 April 2015

#Import needed modules
import numpy as np

class SeparateSources(object):
    """Class that performs BSS using specified method for given observation matrix"""
    
    self.eps = 1.0e-5
    
    def __init__(self,T,q,div_measure,update_rules,**kwargs):
        self.T = T
        self.div_measure = div_measure
        self.update_rules = update_rules
        self.q = q
        #Give some output
        print "Using ",self.div_measure," divergence measure."
        print "Using ",self.update_rules," update rules."
        print "Guessed number of sources ",self.q
        
    
    def minimize_div(self,)
        
    
    def update_uva(self,u,v):
        """Update u,v,A matrices using selected update rules."""
        
        if self.update_rules == 'HALS':
                      
        elif self.update_rules == 'lee_seung_kl':
            
        else:
            raise ValueError("Unknown update rule option.")
        
        return u,v,A
    
    def calculate_div(self,A):
        """Calculate selected divergence measure between observation T and prediction A"""
        
        div = 0.0
        
        if self.div_measure == 'kullback_leibler':
            m,n = self.T.shape
            for i in range(m):
                for j in range(n):
                    if self.T[i,j] == 0.0:
                        term1 = 0.0
                    else:
                        term1 = self.T[i,j]*np.log(self.T[i,j]/A[i,j])
                        
                    term2 = -self.T[i,j]
                    term3 = A[i,j]
                    div = div + (term1 + term2 + term3)
                    
        elif self.div_measure == 'frobenius_norm':
            div = 0.5*np.trace(np.dot(np.transpose(self.T - A),self.T - A))
            
        return div
        
    
    