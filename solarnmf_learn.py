#solarnmf_learn.py

#Will Barnes
#2 April 2015

#Import needed modules
import numpy as np

class SeparateSources(object):
    """Class that performs BSS using specified method for given observation matrix"""
    
    def __init__(self,T,q,div_measure,update_rules,**kwargs):
        self.T = T
        self.div_measure = div_measure
        self.update_rules = update_rules
        self.q = q
        self.ny,self.nx = self.T.shape
        
        self.eps = 1.0e-5
        self.psi = 1.0e-12
        self.max_i = 10000
        self.r = 10
        self.r_iter = 10
        
        print "Using ",self.div_measure," divergence measure."
        print "Using ",self.update_rules," update rules."
        print "Guessed number of sources ",self.q
        
        
    def initialize_uva(self):
        """Initialize the u,v,A matrices by doing a preliminary minimization and using the u,v which give the smallest div."""
        
        u_temp,v_temp = np.random.rand(self.ny,self.q), np.random.rand(self.q,self.nx)
        a_temp = np.dot(u_temp,v_temp)
        u,v,A = u_temp, v_temp, a_temp
        
        div_current = 1.0e+50
        
        for i in range(self.r):
            print "Initialization iteration ",i
            
            u_temp,v_temp,a_temp,div_temp = self.minimize_div(u_temp,v_temp,self.r_iter)
            
            if div_temp[-1] < div_current:
                div_cuurent = div_temp[-1]
                u = u_temp
                v = v_temp
                A = np.dot(u,v)
                
            u_temp,v_temp = np.random.rand(self.ny,self.q), np.random.rand(self.q,self.nx)
            a_temp = np.dot(u_temp,v_temp)
             
        return u,v,A
        
    
    def minimize_div(self,u,v,max_iter):
        """Run the minimization scheme for selected update rules and divergence criteria."""
        
        delta_div = self.eps + 1.0
        div_old = 1.0e+99
        i = 0
        
        div = np.zeros(max_iter)
        
        u = self.normalize_cols(u)
        error = self.T - np.dot(u,v)
        
        while i < max_iter and delta_div > self.eps:
            
            u,v,A = self.update_uva(u,v,error)
            
            div[i] = self.calculate_div(A)
            
            print "At iteration ",i," with divergence ",div[i]
            
            delta_div = np.fabs(div[i] - div_old)
            div_old = div[i]
            i += 1
        
        div = div[0:i]
        
        return u,v,A,div
    
    def update_uva(self,u,v,error):
        """Update u,v,A matrices using selected update rules."""
        
        if self.update_rules == 'HALS':
            
            for i in range(self.q):
                ui = u[:,i]
                vi = v[i,:]
                ti = error + np.outer(ui,vi)
                
                vi = np.dot(np.transpose(ti),ui)/(np.linalg.norm(ui)**2 + self.psi)
                vi[np.where(vi<0.0)] = self.psi
                v[i,:] = vi
                
                ui = np.dot(ti,vi)/(np.linalg.norm(vi)**2 + self.psi)
                ui[np.where(ui<0.0)] = self.psi
                u[:,i] = ui
                
                error = ti - np.outer(ui,vi)
                      
        elif self.update_rules == 'lee_seung_kl':
            
            for i in range(self.q):
                ui = u[:,i]
                vi = v[i,:]
                ti = error + np.outer(ui,vi)
                
                vi = vi*(np.dot(np.transpose(ui),ti/np.dot(ui,vi)))/(np.dot(np.transpose(ui),np.ones((self.ny,self.nx))) + self.psi)
                vi /= np.linalg.norm(vi)**2 + self.psi
                vi[np.where(vi < 0.0)] = self.psi
                v[i,:] = vi
                
                ui = ui*(np.dot(ti/np.dot(ui,vi),np.transpose(vi)))/(np.dot(np.ones((self.ny,self.nx)),np.transpose(vi)) + self.psi)
                ui /= np.linalg.norm(ui)**2 + self.psi
                ui[np.where(ui < 0.0)] = self.psi
                u[:,i] = ui
                
                error = ti - np.outer(ui,vi)
            
        else:
            raise ValueError("Unknown update rule option.")
            
        A = np.dot(u,v)
        
        return u,v,A
    
    def calculate_div(self,A):
        """Calculate selected divergence measure between observation T and prediction A"""
        
        div = 0.0
        
        if self.div_measure == 'kullback_leibler':
            for i in range(self.ny):
                for j in range(self.nx):
                    if self.T[i,j] == 0.0:
                        term1 = 0.0
                    else:
                        term1 = self.T[i,j]*np.log(self.T[i,j]/A[i,j])
                        
                    term2 = -self.T[i,j]
                    term3 = A[i,j]
                    div = div + (term1 + term2 + term3)
                    
        elif self.div_measure == 'frobenius_norm':
            div = 0.5*np.trace(np.dot(np.transpose(self.T - A),self.T - A))
            
        else:
            raise ValueError("Unknown divergence calculation option.")
            
        return div
        
    
    def normalize_cols(self,X):
        r,c = X.shape
        
        for i in range(c):
            xj = X[:,i]
            xj /= np.linalg.norm(xj)
            X[:,i] = xj
            
        return X