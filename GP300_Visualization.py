"""Created on Thu Jun  4 17:33:58 2020.

@author: guepin

Visualize data from ZHAireS simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import hdf5fileinout as hdf5io
import ComputePeak2PeakOnHDF5 as ComputeP2P
import glob

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

PATH_data = '/Users/claireguepin/Projects/GRAND/GP300Outbox/'
# PATH_data = '/Users/claireguepin/Projects/GRAND/TheGP300Outbox/'
PATH_fig = '/Users/claireguepin/Figures/GRAND/'

progenitor = 'Proton'
zenVal = '_'+str(74.8)  # 63.0, 74.8, 81.3, 85.0, 87.1

list_f = glob.glob(PATH_data+'*'+progenitor+'*'+zenVal+'*')
# list_f = glob.glob(PATH_data+'*')
print('Number of files = %i' % (len(list_f)))

energy_tab = np.array([])

# for k in range(0, 10):
for k in range(len(list_f)):
    index = k

    inputfilename = glob.glob(list_f[index] + '/*' + progenitor + '*'
                              + zenVal + '*.hdf5')[0]
    RunInfo = hdf5io.GetRunInfo(inputfilename)
    EventName = hdf5io.GetEventName(RunInfo, 0)
    AntennaInfo = hdf5io.GetAntennaInfo(inputfilename, EventName)
    nantennas = hdf5io.GetNumberOfAntennas(AntennaInfo)
    energy = RunInfo['Energy'][0]
    zenith = 180.-hdf5io.GetEventZenith(RunInfo, 0)
    azimuth = hdf5io.GetEventAzimuth(RunInfo, 0)-180.
    # AntennaID = hdf5io.GetAntennaID(AntennaInfo, 0)
    # efield_loc = hdf5io.GetAntennaEfield(inputfilename, EventName,
    #                                      AntennaID)

    energy_tab = np.append(energy_tab, energy)

    # postab = np.zeros(shape=(nantennas, 3))
    # for i in range(nantennas):
    #     postab[i, 0] = hdf5io.GetAntennaPosition(AntennaInfo, i)[0]
    #     postab[i, 1] = hdf5io.GetAntennaPosition(AntennaInfo, i)[1]
    #     postab[i, 2] = hdf5io.GetAntennaPosition(AntennaInfo, i)[2]

    # # COMPUTE TOTAL PEAK TO PEAK AMPLITUDE
    # p2pE = ComputeP2P.get_p2p_hdf5(inputfilename, antennamax='All',
    #                                antennamin=0, usetrace='efield')
    # p2pE_tot = p2pE[3, :]
    # max_data_test = np.max(p2pE_tot)

    # # All antenna positions
    # antenna_all = np.loadtxt('data/GP300propsedLayout.dat', usecols=(2, 3, 4))

    # antenna_first_num = int(AntennaInfo[0][0][1:])
    # diff_x = antenna_all[antenna_first_num, 0]-postab[0, 0]
    # diff_y = antenna_all[antenna_first_num, 1]-postab[0, 1]
    # diff_z = antenna_all[antenna_first_num, 2]-postab[0, 2]

    # # =============================================================================
    # # VISUALIZE ARRAY

    # plt.figure()
    # ax = plt.gca()

    # plt.scatter(antenna_all[:, 0]/1e3, antenna_all[:, 1]/1e3, 30, color='grey')

    # plt.scatter((postab[:, 0]+diff_x)/1e3, (postab[:, 1]+diff_y)/1e3, 30,
    #             c=p2pE_tot/max_data_test,
    #             cmap=cm.viridis, vmin=0.0, vmax=0.6)

    # ax.xaxis.set_ticks_position('both')
    # ax.yaxis.set_ticks_position('both')
    # ax.tick_params(labelsize=14)

    # # ax.set_xlim([-10, 10])
    # # ax.set_ylim([-10, 10])
    # plt.subplots_adjust(left=0.14)
    # ax.set_xlabel(r"$X\,{\rm\,(km)}$", fontsize=14)
    # ax.set_ylabel(r"$Y\,{\rm\,(km)}$", fontsize=14)
    # ax.axis('equal')

    # # plt.savefig(PATH_fig+'Footprints/GP300_'
    # #             + progenitor+zenVal+'_'+str(index)+'.pdf')

    # plt.show()

# =============================================================================
# VISUALIZE ARRAY

plt.figure()
ax = plt.gca()

plt.hist(energy_tab, 10)
# plt.hist(np.log10(energy_tab), 10)

ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labelsize=14)

# ax.set_xlim([-10, 10])
# ax.set_ylim([-10, 10])
plt.subplots_adjust(left=0.14)
ax.set_xlabel(r"$E,{\rm eV}$", fontsize=14)
# ax.set_xlabel(r"$\log_{10} E$", fontsize=14)
ax.set_ylabel(r"Number", fontsize=14)

# plt.savefig(PATH_fig+'Footprints/GP300_'
#             + progenitor+zenVal+'_'+str(index)+'.pdf')

plt.show()
