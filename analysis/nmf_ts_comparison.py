#nmf_ts_comparison.py

#Will Barnes
#16 April 2015

#Import needed modules
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import argparse
sys.path.append('../')
from solarnmf_observations import MakeData
from solarnmf_learn import SeparateSources

#Parse command line arguments
parser = argparse.ArgumentParser(description='Run NMF method for specific channel over multiple cuts for range of source guesses')
parser.add_argument("-c","--channel",type=int,help="AIA detector channel")
parser.add_argument("-pl","--p_lower",type=int,help="Lower bound for source guess")
parser.add_argument("-pu","--p_upper",type=int,help="Upper bound for source guess")
parser.add_argument("-l","--loop_location",help="AR time series location")
args = parser.parse_args()

#Parent directories
parent_read_file = '/data/datadrive2/AIA_fm_spectra/ts_' + args.loop_location + '/frames(AIA_' + str(args.channel) + ').2D_timeseries'
parent_write_dir = '/data/datadrive2/AIA_fm_spectra/solarnmf_ts_analysis/ts_' + args.loop_location + '/' + str(args.channel) + '/'

#Slice timeseries
N_no_activity = 275
N_ts = 3
ts_cut = []
ts_parent = np.loadtxt(parent_read_file)
ts_parent = ts_parent[N_no_activity:-1]
delta_ts_parent = int(np.round(len(ts_parent)/N_ts))
for i in range(N_ts):
    if i != N_ts - 1:
        ts_cut.append(ts_parent[i*delta_ts_parent:(i+1)*delta_ts_parent])
    else:
        ts_cut.append(ts_parent[i*delta_ts_parent:-1])


#DEBUG
plt.plot(ts_parent)
plt.show()
fig,ax = plt.subplots(1,N_ts)
for i in range(len(ts_cut)):
    ax[i].plot(ts_cut[i])
plt.show()

#Input parameters
#angle = 45.0
#q = range(args.p_lower,args.p_upper+1)


#Declare instance of MakeData class to format input
#data = MakeData('data','timeseries',filename=fn,angle=angle)
#T,Tmat = data.make_t_matrix()
