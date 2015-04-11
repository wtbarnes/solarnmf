#solarnmf_learn.py

#Will Barnes
#2 April 2015

#Import needed modules
import numpy as np

class SeparateSources(object):
    """Class that performs BSS using specified method for given observation matrix"""
    
    def __init__(self,T,q,div_measure,update_rules,**kwargs):
        self.T = T#self.normalize_cols(T)
        self.div_measure = div_measure
        self.update_rules = update_rules
        self.q = q
        self.ny,self.nx = self.T.shape
        
        self.eps = 1.0e-5
        self.psi = 1.0e-12
        self.sparse_u = 0.125
        self.sparse_v = 0.125
        self.reg_a0 = 0.0
        self.reg_tau = 50.0
        self.max_i = 1000
        self.r = 10
        self.r_iter = 10
        
        print "Using ",self.div_measure," divergence measure."
        print "Using ",self.update_rules," update rules."
        print "Guessed number of sources ",self.q
        
        
    def initialize_uva(self):
        """Initialize the u,v,A matrices by doing a preliminary minimization and using the u,v which give the smallest div."""
        
        u_temp,v_temp = np.random.rand(self.ny,self.q), np.random.rand(self.q,self.nx)
        a_temp = np.dot(u_temp,v_temp)
        #u_temp = self.normalize_cols(u_temp)
        
        u,v,A = u_temp, v_temp, a_temp
        
        div_current = 1.0e+50
        
        for i in range(self.r):
            print "Initialization iteration ",i
            
            u_temp,v_temp,a_temp,div_temp = self.minimize_div(u_temp,v_temp,self.r_iter)
            
            if div_temp[-1] < div_current:
                div_current = div_temp[-1]
                u = u_temp
                v = v_temp
                A = np.dot(u,v)
                
            u_temp,v_temp = np.random.rand(self.ny,self.q), np.random.rand(self.q,self.nx)
            a_temp = np.dot(u_temp,v_temp)
            #u_temp = self.normalize_cols(u_temp)
             
        return u,v,A
        
    
    def minimize_div(self,u,v,max_iter):
        """Run the minimization scheme for selected update rules and divergence criteria."""
        
        delta_div = self.eps + 1.0
        div_old = 1.0e+99
        i = 0
        
        div = np.zeros(max_iter)
        
        error = self.T - np.dot(u,v)
        
        while i < max_iter and delta_div > self.eps:
            
            u,v,A = self.update_uva(u,v,i)
            
            div[i] = self.calculate_div(u,v,A,i)
            
            print "At iteration ",i," with divergence ",div[i]
            
            delta_div = np.fabs(div[i] - div_old)
            div_old = div[i]
            i += 1
        
        div = div[0:i]
        
        return u,v,A,div
        
    
    def update_uva(self,u,v,k):
        """Update u,v,A matrices using selected update rules."""
        
        if self.update_rules == 'ALS_reg_sparse':
            
            reg_u,reg_v = self.regularize(k)
            
            vvt_inv = np.linalg.pinv(np.dot(v,np.transpose(v)) + reg_u*np.ones((self.q,self.q)))
            tvt = np.dot(self.T,np.transpose(v)) - self.sparse_u*np.ones((self.ny,self.q))
            u = np.dot(tvt,vvt_inv)
            u[np.where(u<self.psi)] = self.psi
            
            utu_inv = np.linalg.pinv(np.dot(np.transpose(u),u) + reg_v*np.ones((self.q,self.q)))
            utt = np.dot(np.transpose(u),self.T) - self.sparse_v*np.ones((self.q,self.nx))
            v = np.dot(utu_inv,utt)
            v[np.where(v<self.psi)] = self.psi
            
        #elif self.update_rules == 'rHALS':
            
            
                                             
        elif self.update_rules == 'lee_seung_kl':
            
            u = u*np.dot(self.T/np.dot(u,v),np.transpose(v))/(np.dot(np.ones((self.ny,self.nx)),np.transpose(v)) + self.psi)
            v = v*np.dot(np.transpose(u),self.T/np.dot(u,v))/(np.dot(np.transpose(u),np.ones((self.ny,self.nx))) + self.psi)
        
        elif self.update_rules == 'lee_seung_fn':
            
            v = v*np.dot(np.transpose(u),self.T)/(np.dot(np.dot(np.transpose(u),u),v) + self.psi)
            u = u*np.dot(self.T,np.transpose(v))/(np.dot(np.dot(u,v),np.transpose(v)) + self.psi)                          
        
        else:
            raise ValueError("Unknown update rule option.")
            
        #u = self.normalize_cols(u)
            
        A = np.dot(u,v)
        
        return u,v,A
    
    def calculate_div(self,u,v,A,k):
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
                    div += (term1 + term2 + term3)
                    
        elif self.div_measure == 'frobenius_norm':
            div = 0.5*np.linalg.norm(self.T - A,'fro')**2
        
        elif self.div_measure == 'frobenius_norm_reg_sparse':
            reg_u,reg_v = self.regularize(k)
            reg_term_u = reg_u*np.trace(np.dot(np.dot(u,np.ones((self.q,self.q))),np.transpose(u)))
            reg_term_v = reg_v*np.trace(np.dot(np.dot(np.transpose(v),np.ones((self.q,self.q))),v))
            
            div = 0.5*(np.linalg.norm(self.T - A,'fro')**2 + reg_term_u + reg_term_v) + self.sparse_u*u.sum() + self.sparse_v*v.sum()
            
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
        
    def regularize(self,k):
        """Calculate regularization parameter that varies with iteration"""
        reg_u = self.reg_a0*np.exp(-float(k)/self.reg_tau)
        reg_v = self.reg_a0*np.exp(-float(k)/self.reg_tau)
        
        return reg_u,reg_v
        
        