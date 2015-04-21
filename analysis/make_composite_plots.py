#make_composite_plots.py

#Will Barnes
#21 April 2015

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

#Define parent directory and file format
parent_dir = '/data/datadrive2/AIA_fm_spectra/solarnmf_ts_analysis/ts_'+args.loop_location+'/'+str(args.channel)+'/'
fn = 'channel'+str(args.channel)+'_cut%d_q%d'

#Define constant parameters
angle = 45.0
input_type = 'timeseries'
data_option = 'data'
q = np.arange(args.p_lower,args.p_upper+1,1)

#Begin loops
for i in range(args.n_cuts):
    for j in range(len(q)):
        #Read in the data
        #try:
            with open(parent_dir+fn+'.uva'%(i,q[j]),'rb') as f:
                u,v,A,T,Tmat,div = pickle.load(f)
            #Print status
            print "Building plots for cut %d, q=%d"%(i,q[j]) 
            #Declare instance and plot
            plotter = MakeBSSPlots(data_option,input_type,u,v,A,T,div,Tmat=Tmat,angle=angle,ny=T.shape[0],nx=T.shape[0])
            plotter.plot_obs_pred_total_sources_ts(print_fig_filename=parent_dir+'plots/'+fn+'.eps'%(i,q[j]))
        #except:
            #print "Cannot build plot. Incomplete data set."
            #continue