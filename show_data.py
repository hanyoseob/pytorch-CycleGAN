import numpy as np
import matplotlib.pyplot as plt

'''
load data, plot the line profiles
quesiton: what is going on around radius = 2.1?
'''


data_dir = "./database/"

i = 6
M = np.loadtxt( data_dir + "radii_dens_trans_dens_profile_%05d" % i)
fig, ax1 = plt.subplots()
ax1.plot(M[:,0], M[:,1])
ax1.set_xlabel('radius')
ax1.set_ylabel('density')

ax2 = ax1.twinx()
ax2.plot(M[:,0], M[:,2], 'r')
ax2.set_ylabel('transmission')

plt.show()
