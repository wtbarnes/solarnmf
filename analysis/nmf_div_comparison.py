#nmf_div_comparison.py

#Will Barnes
#20 April 2015

#Import needed modules
import sys
import pickle
import argparse
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
sys.path.append('../')
from solarnmf_plotting import MakeBSSPlots

#Parse command line arguments
parser = argparse.ArgumentParser(description='Run NMF method for specific channel over multiple cuts for range of source guesses')
parser.add_argument("-c","--channel",type=int,help="AIA detector channel")
parser.add_argument("-pl","--p_lower",type=int,help="Lower bound for source guess")
parser.add_argument("-pu","--p_upper",type=int,help="Upper bound for source guess")
parser.add_argument("-n","--n_cuts",type=int,help="Number of cuts per channel time series")
parser.add_argument("-l","--loop_location",help="AR time series location")
args = parser.parse_args()

#Define range of possible source guesses
q = np.arange(args.p_lower,args.p_upper+1,1)
eps = 0.1
fs = 18.0
def get_color(i):
    #np.random.seed()
    #r = float(i)/float(args.n_cuts)
    if i == 0:
        color = 'black'
    elif i == 1:
        color = 'blue'
    else:
        color = 'red'
    return color #np.random.rand(3)#[r,r/2.0,r/3.0]

#Define parent directory
parent_dir = '/data/datadrive2/AIA_fm_spectra/solarnmf_ts_analysis/ts_'+args.loop_location+'/'+str(args.channel)+'/'
fn = 'channel'+str(args.channel)+'_cut%d_q%d.uva'

#Begin loop over number of guesses and cuts
q_list = []
div_per_q = []
for i in range(args.n_cuts):
    temp_div = []
    temp_q = []
    for j in range(len(q)):
        print "Processing cut %d, q=%d"%(i,q[j])
        try:
            with open(parent_dir+fn%(i,q[j]),'rb') as f:
                u,v,A,T,Tmat,div = pickle.load(f)
        except:
    	    try:
            	print "Loading incomplete data set for q = %d, cut = %d"%(q[j],i)
            	with open(parent_dir+fn%(i,q[j]),'rb') as f:
                	    u,v,A,div = pickle.load(f)
    	    except:
        		print "Unable to unpickle file."
        		continue
                
        #div_diff = np.where(np.fabs(np.diff(div))<eps)[0]
        #temp_div.append(np.mean(div[div_diff[0]:(div_diff[-1]+1)]))
        temp_div.append(div[-1])
        temp_q.append(q[j])
    
    div_per_q.append(temp_div)
    q_list.append(temp_q)
    

#Exponential fit to our curve
fit_params = []
def exponential_fit(x,a,b):
    return a*np.exp(-x/b)

#Plot divergence as a function of guessed sources for all cuts
fig = plt.figure(figsize=(8,8))
ax = fig.gca()
#ax.set_yscale('log')
ax.set_title(r'SDO/AIA '+str(args.channel)+r' $\AA$, '+args.loop_location,fontsize=fs)
for i in range(args.n_cuts):
    if i == 0:
        lines = ax.plot(q_list[i],div_per_q[i]/np.min(div_per_q[i]),'o',color=get_color(i),label=r'Cut '+str(i))
    else:
        lines += ax.plot(q_list[i],div_per_q[i]/np.min(div_per_q[i]),'o',color=get_color(i),label=r'Cut '+str(i))
    pars,covar = curve_fit(exponential_fit,np.array(q_list[i]),np.array(div_per_q[i]))
    fit_params.append((pars,covar))
    ax.plot(np.array(q_list[i]),exponential_fit(np.array(q_list[i]),*pars)/np.min(div_per_q[i]),'--',color=get_color(i))  
ax.set_ylabel(r'$d/d_{min}$',fontsize=fs)
ax.set_xlabel(r'$k$',fontsize=fs)
ax.set_ylim([.9,4])
ax.set_xlim([args.p_lower-int(args.p_lower/10),args.p_upper+int(args.p_upper/10.0)])
for i in range(args.n_cuts):
    sigma = np.diag(fit_params[i][1])
    tau = fit_params[i][0][1]
    ax.text(35,3.0-0.25*i,r'$\tau_k$ = %3.1f $\pm$ %3.1f'%(tau,sigma[1]),fontsize=18.0,color=get_color(i))
labels = [l.get_label() for l in lines]
ax.legend(lines,labels,loc=1)
plt.savefig('/home/wtb2/Documents/solarnmf/poster/figures/nmf_div_comp_'+str(args.channel)+'.eps',format='eps',dpi=1000)
