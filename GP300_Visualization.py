"""Created on Thu Jun  4 17:33:58 2020.

@author: guepin
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import hdf5fileinout as hdf5io
import ComputePeak2PeakOnHDF5 as ComputeP2P
import glob

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# =============================================================================

PATH_data = '/Users/guepin/Documents/GRAND/TheGP300Outbox/'

progenitor = 'Proton'
zenVal = str(81.3)

list_f = glob.glob(PATH_data+'*'+progenitor+'*'+zenVal+'*')
print('Number of files = %i' % (len(list_f)))

index = 0

inputfilename = glob.glob(list_f[index] + '/*' + progenitor + '*'
                          + zenVal + '*.hdf5')[0]
RunInfo = hdf5io.GetRunInfo(inputfilename)
EventName = hdf5io.GetEventName(RunInfo, 0)
AntennaInfo = hdf5io.GetAntennaInfo(inputfilename, EventName)
nantennas = hdf5io.GetNumberOfAntennas(AntennaInfo)
energy = RunInfo['Energy'][0]
zenith = 180.-hdf5io.GetEventZenith(RunInfo, 0)
azimuth = hdf5io.GetEventAzimuth(RunInfo, 0)-180.

postab = np.zeros(shape=(nantennas, 3))
for i in range(nantennas):
    postab[i, 0] = hdf5io.GetAntennaPosition(AntennaInfo, i)[0]
    postab[i, 1] = hdf5io.GetAntennaPosition(AntennaInfo, i)[1]
    postab[i, 2] = hdf5io.GetAntennaPosition(AntennaInfo, i)[2]

# COMPUTE TOTAL PEAK TO PEAK AMPLITUDE
p2pE = ComputeP2P.get_p2p_hdf5(inputfilename, antennamax='All',
                               antennamin=0, usetrace='efield')
p2pE_tot = p2pE[3, :]
max_data_test = np.max(p2pE_tot)

# =============================================================================
# VISUALIZE ARRAY

plt.figure()
ax = plt.gca()

plt.scatter(postab[:, 0]/1e3, postab[:, 1]/1e3, 30,
            c=p2pE_tot/max_data_test,
            cmap=cm.viridis, vmin=0.0, vmax=0.6)

ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labelsize=14)

# ax.set_xlim([-10, 10])
# ax.set_ylim([-10, 10])
plt.subplots_adjust(left=0.14)
ax.set_xlabel(r"$X\,{\rm\,(km)}$", fontsize=14)
ax.set_ylabel(r"$Y\,{\rm\,(km)}$", fontsize=14)
ax.axis('equal')

plt.show()
