#solarnmf_main_ts.py

#Will Barnes
#31 March 2015

#Import needed modules
import solarnmf_functions as snf
import solarnmf_plot_routines as spr

#Read in and format the time series
results = snf.make_t_matrix("simulation",format="timeseries",nx=100,ny=100,p=10,filename='/home/wtb2/Desktop/gaussian_test.dat')

#Get the dimensions of the T matrix
ny,nx = results['T'].shape

#Set the number of guessed sources
Q = 10

#Initialize the U, V, and A matrices
uva_initial = snf.initialize_uva(nx,ny,Q,10,10,results['T'])

#Start the minimizer
min_results = snf.minimize_div(uva_initial['u'],uva_initial['v'],results['T'],uva_initial['A'],500,1.0e-5)

#Show the initial and final matrices side-by-side
spr.plot_mat_obsVpred(results['T'],min_results['A'])

#Show the initial and final 1d time series curves
spr.plot_ts_obsVpred(results['x'],min_results['A'])

#Show the constituents of the time series on top of the original vector
spr.plot_ts_reconstruction(results['x'],min_results['u'],min_results['v'])

#Show the convergence of the distance metric
spr.plot_convergence(min_results['div'])
