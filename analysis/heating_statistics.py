#heating_statistics.py

#Will Barnes
#22 April 2014

#Import needed modules
import numpy as np
import matplotlib.pyplot as plt

#Parameters
N_fl = 400
time_total = 10800.0
header_lines = 4

#Specify paths and file naming formats
parent_dir = '/data/datadrive2/AIA_fm_spectra/heating_profiles_aia_spectra/'
fn = 'heating_file_L%d.cfg'

#Initialize list of field lines
field_lines = []
#Begin loop over field line number
for i in range(N_fl):
    print "At field line ",i
    #Open file
    f = open(parent_dir+fn%(i+1),'r')
    #Skip the header lines
    for j in range(header_lines):
        f.readline()
    #Read in number of events
    n_e = int(f.readline())
    #Skip a line
    f.readline()
    #Initialize event list
    events = []
    #Read in each heating event
    for j in range(n_e):
        #Read in event parameters
        tmp_list = f.readline().split('\t')
        #Convert to float
        for k in range(len(tmp_list)):
            tmp_list[k] = float(tmp_list[k])
        #Check if event is inside allotted time
        if tmp_list[3] < time_total:
            events.append(tmp_list)
    #Append events to to field line list
    field_lines.append(events)
    #Close the file
    f.close()


#Do some statistics on heating events per field line
events_per_line = []
mean_frequency_per_line = []
for i in field_lines:
    events_per_line.append(len(i))
    mean_frequency_per_line.append(float(len(i))/time_total)

#Statistics for delays between heating events


#Plot histogram of events per line
plt.hist(events_per_line,bins=40)
plt.show()

#Plot histogram of mean frequency per line
plt.hist(mean_frequency_per_line,bins=40)
plt.show()
