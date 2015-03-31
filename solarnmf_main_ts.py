#solarnmf_main_ts.py

#Will Barnes
#31 March 2015

#Import needed modules
import solarnmf_functions as snf
import solarnmf_plot_routines as spr

#Set some initial parameters
P = 10 #simulated number of sources
Q = 10 #guessed number of sources
angle = 45 #rotation angle
nx,ny = 100,100 #dimensions for simulation

#Read in and format the time series
results = snf.make_t_matrix("simulation",format="timeseries",nx=nx,ny=ny,p=P,angle=angle,filename='/home/wtb2/Desktop/gaussian_test.dat')

#Get the dimensions of the T matrix
#ny,nx = results['T'].shape

#Initialize the U, V, and A matrices
uva_initial = snf.initialize_uva(nx,ny,Q,10,10,results['T'])

#Start the minimizer
min_results = snf.minimize_div(uva_initial['u'],uva_initial['v'],results['T'],uva_initial['A'],500,1.0e-5)

#Show the initial and final matrices side-by-side
spr.plot_mat_obsVpred(results['T'],min_results['A'])

#Show the initial and final 1d time series curves
spr.plot_ts_obsVpred(results['x'],min_results['A'],angle=-angle)

#Show the constituents of the time series on top of the original vector
spr.plot_ts_reconstruction(results['x'],min_results['u'],min_results['v'],angle=-angle)

#Show the convergence of the distance metric
spr.plot_convergence(min_results['div'])
