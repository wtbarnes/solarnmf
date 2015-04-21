#nmf_div_comparison.py

#Will Barnes
#20 April 2015

#Import needed modules
import sys
import pickle
import argparse
import numpy as np
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
eps = 10.0
fs = 18.0
def get_color(i):
    r = float(i)/float(len(q))
    return [r,r/2.0,r/3.0]

#Define parent directory
parent_dir = '/data/datadrive2/AIA_fm_spectra/solarnmf_ts_analysis/ts_'+args.loop_location+'/'+str(args.channel)+'/'
fn = 'channel'+str(args.channel)+'_cut%d_q%d.uva'

#Begin loop over number of guesses and cuts
div_per_q = []
for i in range(args.n_cuts):
    temp_div = []
    for j in range(len(q)):
        try:
            with open(parent_dir+fn%(i,q[j]),'rb') as f:
                u,v,A,T,Tmat,div = pickle.load(f)
        except:
            print "Loading incomplete data set for q = %d, cut = %d"%(q[j],i)
            with open(parent_dir+fn%(i,q[j]),'rb') as f:
                u,v,A,div = pickle.load(f)
                
        div_diff = np.where(np.fabs(np.diff(div))<eps)
        temp_div.append(np.mean(div[div_diff[0]:div_diff[-1]+1]))
    
    div_per_q.append(temp_div)
    

#Plot divergence as a function of guessed sources for all cuts
lines = 0
fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_title(r'SDO/AIA '+str(args.channel)+r' $\AA$, '+args.loop_location,fontsize=fs)
for i in range(args.n_cuts):
    lines += ax.plot(div_per_q,color=get_color(i),label=r'Cute '+str(i))
ax.set_ylabel(r'$d/d_{min}$',fontsize=fs)
ax.set_xlabel(r'$q$',fontsize=fs)
labels = [l.get_label() for l in lines]
ax.legend(lines,labels,loc=1)
plt.show()
