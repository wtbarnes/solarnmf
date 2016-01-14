#solarnmf_learn.py

#Will Barnes
#2 April 2015

#Import needed modules
import numpy as np
import pickle
import logging
from scipy.linalg import toeplitz

class SeparateSources(object):
    """Class that performs BSS using specified method for given observation matrix"""

    def __init__(self,T,q,params,**kwargs):
        self.T = T#self.normalize_cols(T)
        self.q = q
        self.ny,self.nx = self.T.shape

        self.div_measure = params['div_measure']
        self.update_rules = params['update_rules']
        self.eps = params['eps']
        self.psi = params['psi']
        self.sparse_u = params['sparse_u']
        self.sparse_v = params['sparse_v']
        self.reg_0 = params['reg_0']
        self.reg_tau = params['reg_tau']
        self.max_i = params['max_i']
        self.r = params['r']
        self.r_iter = params['r_iter']
        
        self.lambda_1 = params['lambda_1']
        self.lambda_2 = params['lambda_2']
        self.alpha = params['alpha']
        self.beta = 1.0 - self.alpha
        self.l_toeplitz = params['l_toeplitz']
            
        #Configure logger
        self.logger = logging.getLogger(type(self).__name__)
            
        if 'print_results' in kwargs:
            self.print_results = kwargs['print_results']
        else:
            self.print_results = False
        
        #Log setup options
        self.logger.info("Using "+self.div_measure+" divergence measure.")
        self.logger.info("Using "+self.update_rules+" update rules.")
        self.logger.info("Guessed number of sources "+str(self.q))


    def initialize_uva(self):
        """Initialize the u,v,A matrices by doing a preliminary minimization and using the u,v which give the smallest div."""

        u_temp,v_temp = np.random.rand(self.ny,self.q), np.random.rand(self.q,self.nx)
        a_temp = np.dot(u_temp,v_temp)
        #u_temp = self.normalize_cols(u_temp)

        u,v,A = u_temp, v_temp, a_temp

        div_current = 1.0e+50

        for i in range(self.r):
            
            self.logger.info("Initialization iteration "+str(i))

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
            
            if i%10 == 0:
                self.logger.info("At iteration "+str(i)+" with divergence "+str(div[i]))
                if (self.print_results is not False) and (max_iter > self.r_iter):
                    self.file_io(u,v,A,div)

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

        elif self.update_rules == 'chen_cichocki_reg_sparse':
            
            Q = 1.0/self.nx*np.dot(np.transpose(self.eye_minus_toeplitz()),self.eye_minus_toeplitz())
            A = np.dot(u,v)
            
            def update_u(u,v,A):
                for i in range(self.ny):
                    for j in range(self.q):
                        numer = 0.0
                        for t in range(self.nx):
                            numer += v[j,t]*self.T[i,t]/A[i,t]
                        u[i,j] = u[i,j]*numer/(np.sum(v[j,:]))
                u[np.where(u<self.psi)] = self.psi
                return u
                
            def update_v(u,v,A,Q):
                utt = np.dot(np.transpose(u),self.T)
                uta_vq  = np.dot(np.transpose(u),A) + self.lambda_1*np.dot(v,Q)
                for j in range(self.q):
                    for t in range(self.nx):
                        v[j,t] = v[j,t]*utt[j,t]/(uta_vq[j,t] + self.lambda_2/self.nx*(np.sum(v[:,t]) - 2.0*v[j,t]))
                v[np.where(v<self.psi)] = self.psi
                return v
                
            if k%2 == 0:
                u = update_u(u,v,A)
                A = np.dot(u,v)
                v = update_v(u,v,A,Q)
                A = np.dot(u,v)
            else:
                v = update_v(u,v,A,Q)
                A = np.dot(u,v)
                u = update_u(u,v,A)
                A = np.dot(u,v)

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
            
        elif self.div_measure == 'multiplicative_reg_sparse':
            
            term1 = np.sum((self.T - A)**2)
            
            term2 = 0.0
            emt = self.eye_minus_toeplitz()
            for i in range(self.q):
                term2 += np.linalg.norm(np.dot(emt,np.transpose(v[i,:])))**2
                
            term3 = 2.0*np.sum(np.dot(v,np.transpose(v))) - 3.0*np.trace(np.dot(v,np.transpose(v)))
            
            div = term1 + self.lambda_1/self.nx*term2 + self.lambda_2/(2.0*self.nx)*term3

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
        reg_u = self.reg_0*np.exp(-float(k)/self.reg_tau)
        reg_v = self.reg_0*np.exp(-float(k)/self.reg_tau)

        return reg_u,reg_v
        
    def eye_minus_toeplitz(self):
        
        rv,cv = np.zeros(self.nx),np.zeros(self.nx)
        
        rv[0] = self.beta
        
        for i in range(self.l_toeplitz):
            cv[i] = (self.alpha**i)*self.beta
        
        return np.eye(self.nx) - toeplitz(cv,rv)
        
        
    def file_io(self,u,v,A,div):
        with open(self.print_results,'wb') as f:
            pickle.dump([u,v,A,div],f)
        f.close()
        
