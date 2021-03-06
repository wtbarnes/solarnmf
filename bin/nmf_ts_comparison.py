#nmf_ts_comparison.py

#Will Barnes
#16 April 2015

#Import needed modules
import numpy as np
import pickle
import datetime
import logging
import sys
import argparse
import multiprocessing
#Import solarnmf classes
sys.path.append('/home/wtb2/Documents/solarnmf/')
from solarnmf_observations import MakeData
from solarnmf_learn import SeparateSources

#Declare function for minimization process
def minimizer_worker(Tmat,T,q,params,i_cut,top_dir,channel):
    out_file_prefix = top_dir+'channel'+str(channel)+'_cut'+str(i_cut)+'_q'+str(q)
    #Configure logging
    logging.basicConfig(filename=out_file_prefix+'.log',level=logging.INFO)
    logging.info('worker logger -- channel: '+str(channel)+', cut = '+str(i_cut)+', q = '+str(q))
    #Start minimizer
    minimizer = SeparateSources(Tmat,q,params,verbose=True,logger=True,print_results=out_file_prefix+'.uva')
    u_i,v_i,A_i = minimizer.initialize_uva()
    u,v,A,div = minimizer.minimize_div(u_i,v_i,minimizer.max_i)
    #Finish log
    logging.info('run finished')
    #Save data
    with open(out_file_prefix+'.uva','wb') as f:
        pickle.dump([u,v,A,T,Tmat,div],f)
    f.close()


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
    ts_cut[i]/np.max(ts_cut[i])

#Input parameters
angle = 45.0
q = range(args.p_lower,args.p_upper+1)
N_q = len(q)

#Open log file
logger = open(parent_write_dir+'channel_'+str(args.channel)+'.log','w')
logger.write('solarnmf_analysis logger -- channel '+str(args.channel)+'\n')
logger.write('Starting run at:'+str(datetime.datetime.now())+'\n')

#Set parameters for the minimization
params = {'eps':1.0e-3,'psi':1.0e-16,'sparse_u':0.125,'sparse_v':0.125,'reg_0':20.0,'reg_tau':50.0,'max_i':150,'r':10,'r_iter':5}
params['lambda_1'] = 0.0001
params['lambda_2'] = 0.0001
params['alpha'] = 0.8
params['l_toeplitz'] = 5
params['div_measure'] = 'multiplicative_reg_sparse'
params['update_rules'] = 'chen_cichocki_reg_sparse'

#Begin iteration over q and N_ts
for i in range(N_ts):
    #Declare instance of MakeData class to format input
    data = MakeData('data','timeseries',file=ts_cut[i],angle=angle)
    T,Tmat = data.make_t_matrix()

    #Write to the logger
    logger.write('Starting runs for cut '+str(i)+' at '+str(datetime.datetime.now())+'\n')

    for j in range(N_q):
        #Write to log file
        logger.write('Running minimizer for q = '+str(q[j])+' for cut '+str(i)+'\n')
        #Start process
        mtp = multiprocessing.Process(target=minimizer_worker,args=(Tmat,T,q[j],params,i,parent_write_dir,args.channel))
        mtp.start()


#Close the logger
logger.close()
