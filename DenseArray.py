#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:37:04 2020.

@author: guepin
"""

# =============================================================================
# GRAND MACHINE LEARNING FOR RECONSTRUCTION
# Goal: study energy reconstruction with convolutional neural network
# Data: ZHAireS, flat dense array 13*25, 125m spacing
# Contents: see FLAGS and CHOOSE PARAMETERS
# =============================================================================

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import glob
import hdf5fileinout as hdf5io
import ComputePeak2PeakOnHDF5 as ComputeP2P
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pickle
import time
import random

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

manualSeed = 1
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

# =============================================================================

# FLAGS

# Visualize footprint and signal for one event
vis_data = 0

# Create data set for one progenitor, all showers
create_fulldataset = 0
# Create data set for one progenitor, one zenith angle
create_dataset = 0

# Save important data set quantities
quantities_databank = 0

# Deep learning training and test
train_model = 0
test_model = 0

# Visualization tool
tensor_board = 0

# =============================================================================

# CHOOSE PARAMETERS FOR VISUALIZE DATA, CREATE DATASET, TRAIN OR TEST

# Choose progenitor: 'Proton'
progenitor = 'Proton'

# Choose zenith angle: '56.1', '62.5' or '72.4' (other values available)
ZenVal = '56.1'

# Choose signal: 'efield', 'voltage', 'filteredvoltage', 'filteredvoltagenoise'
trace = 'filteredvoltage'

# Choose labels to be predicted by the network: 'energy' or 'energy_azimuth'
net_labels = 'energy'

# Choose number of epochs for training
n_epochs = 200

# Choose learning rate, scheduler 'cst' or 'dec'
learn_rate = 1e-3
lr_scheduler = 'cst'

# Choose weight decay
wd = 0.0

# Choose batch size: 1 fully stochastic gradient descent
batchsize = 1

# Path for data bank (ZHAireS simulations)
# Outbox, OutboxVoltage, OutboxSignalProc
PATH_data = '/Users/guepin/Documents/GRAND/OutboxSignalProc/'

# Name with chosen properties for saving information and figures
name_prop = net_labels+'_'+progenitor+'_zen'+ZenVal+'_'+trace+'_lr'\
    + str(learn_rate)+'_'+lr_scheduler+'_wd'+str(wd)+'_bs'+str(batchsize)\
    + '_nepoch'+str(n_epochs)

# Path to save trained network properties
PATH = './data/net_'+name_prop+'.pth'

# Path to save figures
PATH_fig = '/Users/guepin/Documents/NotesLatex/GRAND_reco/'

# Maximum peak to peak amplitude, to normalize the data
P2PMax = pickle.load(open('ZhairesSimulations_Parameters_' +
                          progenitor+'.p', "rb"))[-1]

# =============================================================================

# NETWORK


class Net(nn.Module):
    """Define the network."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 4)
        self.conv2 = nn.Conv2d(4, 2, 2)
        # self.conv2 = nn.Conv2d(4, 4, 4)
        # Default value of stride is kernel_size
        self.pool1 = nn.MaxPool2d(2)
        # self.pool1 = nn.MaxPool2d(2, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(2*2*5, 2*2*5)
        # self.fc3 = nn.Linear(2*2*5, 2*2*5)
        # self.fc1 = nn.Linear(4*3*9, 2*2*5)
        if net_labels == 'energy':
            self.fc2 = nn.Linear(2*2*5, 1)
        elif net_labels == 'energy_azimuth':
            self.fc2 = nn.Linear(2*2*5, 2)
        else:
            print('ERROR: CHOOSE LABELS!')

    def forward(self, x):
        """Forward propagation."""
        # print(np.shape(x))
        x = self.pool1(F.relu(self.conv1(x)))
        # print(np.shape(x))
        x = self.pool2(F.relu(self.conv2(x)))
        # print(np.shape(x))
        x = x.view(-1, 2*2*5)
        # x = x.view(-1, 4*3*9)
        # print(np.shape(x))
        x = F.relu(self.fc1(x))
        # print(np.shape(x))
        # x = self.fc3(x)
        x = self.fc2(x)
        # print(np.shape(x))
        return x


net = Net()
# tensor_x = torch.load('data/tensor_x_p2p_'+net_labels+'_'
#                       + progenitor+'_zen'+ZenVal+'_'+trace+'.pt') / P2PMax
# net(tensor_x[0:1])

# Loss function: Mean Square Error Loss
criterion = nn.MSELoss()

# Optimizer: Adam method
optimizer = optim.Adam(net.parameters(), lr=learn_rate, betas=(0.9, 0.999),
                       eps=1e-08, weight_decay=wd, amsgrad=False)

lmbda = lambda epoch: 0.99
scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)


# Accuracy
def accuracy_model(model, real_labels, pred_labels, maxlogdiff):
    """Test the accuracy of the energy reconstruction."""
    n_items = len(pred_labels)
    print(n_items)
    n_correct = torch.sum((torch.abs(real_labels - pred_labels) < maxlogdiff))
    result = (n_correct.item() * 100.0 / n_items)
    return result


def count_parameters(model):
    """Count trainable parameters in the network."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('***********************************')
print('*** NETWORK AND DATA PROPERTIES ***')
print("Number of trainable parameters in the model: %i"
      % count_parameters(net))
print('Number of epochs: %i' % n_epochs)
print('Learning rate = %.2e' % learn_rate)
print('Batch size = %i' % batchsize)
print('Progenitor: %s' % progenitor)
print('Zenith = %sº' % ZenVal)
print('Trace: %s' % trace)
print('Labels: %s' % net_labels)
print('***********************************')


# =============================================================================

# VISUALIZE DATA

if vis_data:

    # IMPORT FILES, CHOOSE ONE ZENITH AND ONE PRIMARY
    list_f = glob.glob(PATH_data+'*'+ZenVal+'*'+progenitor+'*')
    print('Number of files = %i' % (len(list_f)))

    # CHOOSE INPUT FILE FOR EXAMPLE
    # index_f = 2
    # inputfilename = glob.glob(list_f[index_f] + '/*' + ZenVal + '*' +
    #                           progenitor + '*.hdf5')[0]
    inputfilename = '/Users/guepin/Documents/GRAND/OutboxSignalProc/'\
        + '125m_0.133_56.1_0_Proton_100_04/'\
        + '125m_0.133_56.1_0_Proton_100_04.hdf5'
    RunInfo = hdf5io.GetRunInfo(inputfilename)
    EventName = hdf5io.GetEventName(RunInfo, 0)
    AntennaInfo = hdf5io.GetAntennaInfo(inputfilename, EventName)
    nantennas = hdf5io.GetNumberOfAntennas(AntennaInfo)

    postab = np.zeros(shape=(nantennas, 3))
    for i in range(nantennas):
        postab[i, 0] = hdf5io.GetAntennaPosition(AntennaInfo, i)[0]
        postab[i, 1] = hdf5io.GetAntennaPosition(AntennaInfo, i)[1]
        postab[i, 2] = hdf5io.GetAntennaPosition(AntennaInfo, i)[2]

    # COMPUTE TOTAL PEAK TO PEAK AMPLITUDE
    # In the future compare efield, voltage and filteredvoltage (usetrace=)
    p2pE = ComputeP2P.get_p2p_hdf5(inputfilename, antennamax='All',
                                   antennamin=0, usetrace='efield')
    p2pE_tot = p2pE[3, :]
    max_data_test = np.max(p2pE_tot)

    posX = np.zeros(shape=(13, 25))
    posY = np.zeros(shape=(13, 25))
    p2p = np.zeros(shape=(13, 25))
    for l in range(13):
        for c in range(25):
            index = c*13+(12-l)
            posX[l, c] = postab[index, 0]
            posY[l, c] = postab[index, 1]
            p2p[l, c] = p2pE_tot[index]

    # =============================================================================

    # PLOT ARRAY

    plt.figure()
    ax = plt.gca()

    plt.scatter(postab[:, 0]/1e3, postab[:, 1]/1e3, 20,
                c=p2pE_tot/max_data_test,
                cmap=cm.viridis, vmin=0., vmax=1.)

    plt.scatter(posX/1e3, posY/1e3, c=p2p)

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labelsize=14)

    # ax.set_xlim([-10000,10000])
    # ax.set_ylim([-10000,10000])
    plt.subplots_adjust(left=0.14)
    ax.set_xlabel(r"$X\,{\rm\,(km)}$", fontsize=14)
    ax.set_ylabel(r"$Y\,{\rm\,(km)}$", fontsize=14)
    ax.axis('equal')

    plt.show()

    # =============================================================================

    # PLOT SIGNAL
    # Example for one antenna

    ant_num = 162  # 162, 250

    efield_tab = hdf5io.GetAntennaEfield(inputfilename, EventName, ant_num)
    volt_tab = hdf5io.GetAntennaVoltage(inputfilename, EventName, ant_num)
    filtvolt_tab = hdf5io.GetAntennaFilteredVoltage(inputfilename, EventName,
                                                    ant_num)
    filtvoltnoise_tab = hdf5io.GetAntennaFilteredVoltageNoise(inputfilename,
                                                              EventName,
                                                              ant_num)
    maxi_efield = np.max(np.abs(efield_tab[:, 1:4]))
    maxi_volt = np.max(np.abs(volt_tab[:, 1:4]))
    maxi_filtvolt = np.max(np.abs(filtvolt_tab[:, 1:4]))
    maxi_sigproc = np.max(np.abs(filtvoltnoise_tab[:, 1:4]))
    maxi = np.max([maxi_efield, maxi_volt, maxi_filtvolt])

    plt.figure()
    ax = plt.gca()

    plt.plot(efield_tab[:, 0], efield_tab[:, 1]/maxi_efield)
    plt.plot(efield_tab[:, 0], efield_tab[:, 2]/maxi_efield)
    plt.plot(efield_tab[:, 0], efield_tab[:, 3]/maxi_efield)

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labelsize=14)

    # ax.set_xlim([-50, 150])
    ax.set_ylim([-1, 1])

    plt.show()

    plt.figure()
    ax = plt.gca()

    plt.plot(volt_tab[:, 0], volt_tab[:, 1]/maxi_volt)
    plt.plot(volt_tab[:, 0], volt_tab[:, 2]/maxi_volt)
    plt.plot(volt_tab[:, 0], volt_tab[:, 3]/maxi_volt)

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labelsize=14)

    # ax.set_xlim([-50, 150])
    ax.set_ylim([-1, 1])

    plt.show()

    plt.figure()
    ax = plt.gca()

    plt.plot(filtvolt_tab[:, 0], filtvolt_tab[:, 1]/maxi_filtvolt)
    plt.plot(filtvolt_tab[:, 0], filtvolt_tab[:, 2]/maxi_filtvolt)
    plt.plot(filtvolt_tab[:, 0], filtvolt_tab[:, 3]/maxi_filtvolt)

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labelsize=14)

    # ax.set_xlim([-50, 150])
    ax.set_ylim([-1, 1])

    plt.show()

    plt.figure()
    ax = plt.gca()

    plt.subplots_adjust(left=0.13)
    
    plt.plot(filtvoltnoise_tab[:, 0], filtvoltnoise_tab[:, 1]/maxi_sigproc)
    plt.plot(filtvoltnoise_tab[:, 0], filtvoltnoise_tab[:, 2]/maxi_sigproc)
    plt.plot(filtvoltnoise_tab[:, 0], filtvoltnoise_tab[:, 3]/maxi_sigproc)

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labelsize=14)

    plt.xlabel(r'Time (ns)', fontsize=14)
    plt.ylabel(r'Filtered signal with Gaussian noise, normalized', fontsize=14)

    # ax.set_xlim([-50, 150])
    # ax.set_ylim([-1, 1])

    plt.show()

# =============================================================================

# CREATE DATASET for ML

# Full data set: all energies and zenith angles for one progenitor
if create_fulldataset:

    p2p_array = []
    energy = []
    zenith = []

    list_f = glob.glob(PATH_data+'*'+progenitor+'*')
    print('Number of files = %i' % (len(list_f)))
    num_count = 0

    for k in range(len(list_f)):
        index_f = k
        inputfilename = glob.glob(list_f[index_f]+'/*'+progenitor+'*.hdf5')
        if len(inputfilename) > 0:
            num_count += 1
            inputfilename = inputfilename[0]
            RunInfo = hdf5io.GetRunInfo(inputfilename)
            Zenith = 180.-hdf5io.GetEventZenith(RunInfo, 0)
            Azimuth = 360.-hdf5io.GetEventAzimuth(RunInfo, 0)

            p2pE = ComputeP2P.get_p2p_hdf5(inputfilename, antennamax='All',
                                           antennamin=0, usetrace='efield')
            p2pE_tot = p2pE[3, :]

            # Reshape
            p2p = np.zeros(shape=(13, 25))
            for l in range(13):
                for c in range(25):
                    index = c * 13 + (12-l)
                    p2p[l, c] = p2pE_tot[index]

            p2p_array.append(np.array([p2p]))

            energy.append(RunInfo['Energy'][0])
            zenith.append(Zenith)

    # Transform to torch tensors
    tensor_x = torch.Tensor(p2p_array)
    tensor_y = torch.Tensor(np.array([energy, zenith]))

    print('Number of files counted = %i' % num_count)

    torch.save(tensor_x, 'data/\
               tensor_x_p2p_energy_zenith_'+progenitor+'_'+trace+'.pt')
    torch.save(tensor_y, 'data/\
               tensor_y_p2p_energy_zenith_'+progenitor+'_'+trace+'.pt')

# Specific data set: all energies for one zenith angle and one progenitor
if create_dataset:

    p2p_array = []
    params = []

    list_f = glob.glob(PATH_data+'*' + ZenVal
                       + '*' + progenitor + '*')
    print('Number of files = %i' % (len(list_f)))

    for k in range(len(list_f)):
        index_f = k
        inputfilename = glob.glob(list_f[index_f] + '/*' + ZenVal + '*'
                                  + progenitor + '*.hdf5')
        if len(inputfilename) > 0:
            inputfilename = inputfilename[0]
            RunInfo = hdf5io.GetRunInfo(inputfilename)
            Zenith = 180.-hdf5io.GetEventZenith(RunInfo, 0)
            Azimuth = 360.-hdf5io.GetEventAzimuth(RunInfo, 0)

            p2pE = ComputeP2P.get_p2p_hdf5(inputfilename, antennamax='All',
                                           antennamin=0, usetrace=trace)
            p2pE_tot = p2pE[3, :]

            # Reshape
            p2p = np.zeros(shape=(13, 25))
            for l in range(13):
                for c in range(25):
                    index = c * 13 + (12-l)
                    p2p[l, c] = p2pE_tot[index]

            p2p_array.append(np.array([p2p]))

            if net_labels == 'energy':
                params.append(np.array([RunInfo['Energy'][0]]))
            elif net_labels == 'energy_azimuth':
                params.append(np.array([RunInfo['Energy'][0], Azimuth]))
            else:
                print('ERROR: CHOOSE LABELS!')

    # max_p2p=np.max(p2p_array)

    # Transform to torch tensors
    tensor_x = torch.Tensor(p2p_array)
    tensor_y = torch.Tensor(params)

    torch.save(tensor_x, 'data/tensor_x_p2p_'+net_labels+'_'+progenitor
               + '_zen'+ZenVal+'_'+trace+'.pt')
    torch.save(tensor_y, 'data/tensor_y_p2p_'+net_labels+'_'+progenitor
               + '_zen'+ZenVal+'_'+trace+'.pt')

# =============================================================================

# IMPORTANT QUANTITIES FROM THE FULL DATA BANK

if quantities_databank:

    tensor_x = torch.load('data/tensor_x_p2p_energy_zenith_'
                          + progenitor+'_efield.pt')
    tensor_y = torch.load('data/tensor_y_p2p_energy_zenith_'
                          + progenitor+'_efield.pt')

    num_sim, dummy, dim_array_l, dim_array_c = np.array(np.shape(tensor_x))
    E_min = np.min(np.array(tensor_y[0, :]))
    E_max = np.max(np.array(tensor_y[0, :]))
    zenith_min = np.min(np.array(tensor_y[1, :]))
    zenith_max = np.max(np.array(tensor_y[1, :]))

    max_P2P = np.zeros(num_sim)
    for i in range(num_sim):
        max_P2P[i] = np.max(np.array(tensor_x[i, :]))

    amplitudeP2Ptot_min = np.min(np.array(tensor_x))
    amplitudeP2Ptot_max = np.max(np.array(tensor_x))

    print("num_sim = %i" % num_sim)
    print("E_min = %.2e EeV" % E_min)
    print("E_max = %.2e EeV" % E_max)
    print("zenith_min = %.2f deg" % zenith_min)
    print("zenith_max = %.2f deg" % zenith_max)
    print("amplitude P2P tot min = %.2e (unit?)" % amplitudeP2Ptot_min)
    print("amplitude P2P tot max = %.2e (unit?)" % amplitudeP2Ptot_max)
    print("amplitude P2P max min = %.2e (unit?)" % np.min(max_P2P))
    print("amplitude P2P max max = %.2e (unit?)" % np.max(max_P2P))

    filename = 'ZhairesSimulations_Parameters_'+progenitor+'.p'
    info_sim = [num_sim, dim_array_l, dim_array_c, E_min, E_max,
                zenith_min, zenith_max, np.min(max_P2P), np.max(max_P2P)]
    f = open(filename, "wb")
    pickle.dump(info_sim, f)
    f.close()

# =============================================================================

# TRAIN MODEL

if train_model:

    time_ini = time.time()

    tensor_x = torch.load('data/tensor_x_p2p_'+net_labels+'_'
                          + progenitor+'_zen'+ZenVal+'_'+trace+'.pt') / P2PMax
    tensor_y = torch.load('data/tensor_y_p2p_'+net_labels+'_'
                          + progenitor+'_zen'+ZenVal+'_'+trace+'.pt')

    tensor_y[:, 0] = np.log10(tensor_y[:, 0])
    if len(tensor_y[0, :]) > 1:
        tensor_y[:, 1] /= 360.

    # Create dataset
    my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)

    train_length = tensor_x.shape[0]-100
    test_length = 100
    lengths = [train_length, test_length]
    train_dataset, test_dataset = torch.utils.data.random_split(my_dataset,
                                                                lengths)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                                   batch_size=batchsize)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True)

    # SAVE TRAIN AND TEST DATALOADERS
    torch.save(train_dataloader, 'data/train_dataloader_'
               + name_prop+'.pt')
    torch.save(test_dataloader, 'data/test_dataloader_'
               + name_prop+'.pt')

    loss_cumul_arr = []
    loss_arr = []

    print("Start Training")

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0

        # print("Epoch: %i" %epoch)

        for i, data in enumerate(train_dataloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # save statistics
            running_loss += loss.item()
            loss_arr.append(loss.item())

        loss_cumul_arr.append(running_loss)

        if lr_scheduler != 'cst':
            scheduler.step()

    time_end = time.time()

    print('End Training')
    print('Time Training: %.2f s' % (time_end-time_ini))

    # Save trained model
    torch.save(net.state_dict(), PATH)

    # Calculate the accuracy of training
    solu = []
    pred = []
    with torch.no_grad():
        for data in train_dataloader:
            images, labels = data
            outputs = net(images)
            for k in range(len(labels)):
                solu.append(labels[k][0].item())
                pred.append(outputs[k][0].item())

    accuracy_train = accuracy_model(net, torch.Tensor(solu),
                                    torch.Tensor(pred), 0.1)
    print('Accuracy train: %0.2f %%' % (accuracy_train))

    # Compare the accuracy for low and high energies
    meanlog10E = np.mean(np.array(tensor_y[:, 0]))
    solu_low = []
    solu_high = []
    pred_low = []
    pred_high = []
    with torch.no_grad():
        for data in train_dataloader:
            images, labels = data
            outputs = net(images)
            for k in range(len(labels)):
                if labels[k][0] <= meanlog10E:
                    solu_low.append(labels[k][0].item())
                    pred_low.append(outputs[k][0].item())
                else:
                    solu_high.append(labels[k][0].item())
                    pred_high.append(outputs[k][0].item())

    accuracy_train_low = accuracy_model(net, torch.Tensor(solu_low),
                                        torch.Tensor(pred_low), 0.1)
    accuracy_train_high = accuracy_model(net, torch.Tensor(solu_high),
                                         torch.Tensor(pred_high), 0.1)
    print('Accuracy train low E: %0.2f %%' % (accuracy_train_low))
    print('Accuracy train high E: %0.2f %%' % (accuracy_train_high))

    # Save these results in a pickle file
    filename = 'results/ResultsTrain_'+name_prop+'.p'
    acc_tab = ['accuracy train', accuracy_train, 'accuracy train low',
               accuracy_train_low, 'accuracy train high', accuracy_train_high]
    f = open(filename, "wb")
    pickle.dump(acc_tab, f)
    pickle.dump(solu, f)
    pickle.dump(pred, f)
    f.close()

    # =============================================================================
    # FIGURES
    # =============================================================================

    fig = plt.figure()
    ax = plt.gca()
    plt.plot(loss_cumul_arr, linewidth=2)
    plt.xlabel(r'Number of epochs', fontsize=16)
    plt.ylabel(r'Cumulative loss', fontsize=16)
    ax.set_xlim([0, len(loss_cumul_arr)])
    ax.set_ylim([0, 100.])
    ax.tick_params(labelsize=14)
    plt.savefig(PATH_fig+'CumulLoss_'+name_prop+'.pdf')
    plt.show()

    # =============================================================================

    fig = plt.figure()
    ax = plt.gca()
    plt.plot(loss_arr, linewidth=2)
    plt.xlabel(r'Number of iterations', fontsize=16)
    plt.ylabel(r'Loss', fontsize=16)
    ax.set_xlim([0, len(loss_arr)])
    ax.set_ylim([0, 1.])
    ax.tick_params(labelsize=14)
    plt.savefig(PATH_fig+'Loss_'+name_prop+'.pdf')
    plt.show()

    # =============================================================================

    fig = plt.figure()
    ax = plt.gca()
    y, x, _ = plt.hist(np.array(pred) - np.array(solu),
                       bins=np.arange(-0.5, 0.5 + 0.05, 0.05))
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labelsize=14)
    mean_train = np.mean(np.array(pred)-np.array(solu))
    std_train = np.std(np.array(pred)-np.array(solu))
    # ax.set_xlim([-abs(x).max(), abs(x).max()])
    ax.set_xlim([-0.5, 0.5])
    plt.xlabel(r'$\log_{10} (E_{\rm pred})-\log_{10} (E_{\rm real})$',
               fontsize=14)
    plt.ylabel(r'$N$', fontsize=14)
    plt.text(abs(x).max()/3, y.max()-10,
             r'$\rm Mean = {0:.4f}$'.format(mean_train), fontsize=14)
    plt.text(abs(x).max()/3, y.max()-10-y.max()/10,
             r'$\rm Std = {0:.4f}$'.format(std_train), fontsize=14)
    plt.savefig(PATH_fig+'HistTrain_'+name_prop+'.pdf')
    plt.show()

# =============================================================================

# TEST MODEL

if test_model:

    net.load_state_dict(torch.load(PATH))
    net.eval()

    test_dataloader = torch.load('data/test_dataloader_'
                                 + name_prop+'.pt')

    solu = []
    pred = []
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = net(images)
            for k in range(len(labels)):
                solu.append(labels[k][0].item())
                pred.append(outputs[k][0].item())

    accuracy_test = accuracy_model(net, torch.Tensor(solu),
                                   torch.Tensor(pred), 0.1)
    print('Accuracy test: %0.2f %%' % (accuracy_test))

    filename = 'results/ResultsTest_'+name_prop+'.p'
    acc_tab = ['accuracy test', accuracy_test]
    f = open(filename, "wb")
    pickle.dump(acc_tab, f)
    pickle.dump(solu, f)
    pickle.dump(pred, f)
    f.close()

    # =============================================================================
    # FIGURES
    # =============================================================================

    fig = plt.figure()
    ax = plt.gca()

    plt.plot(solu, linestyle='', marker='o')
    plt.plot(pred, linestyle='', marker='x')

    ax.tick_params(labelsize=14)

    # ax.set_xlim([-10000,10000])
    # ax.set_ylim([-10000,10000])
    # ax.axis('equal')
    plt.show()

    # =============================================================================

    fig = plt.figure()
    ax = plt.gca()
    y, x, _ = plt.hist(np.array(pred) - np.array(solu),
                       bins=np.arange(-0.5, 0.5 + 0.05, 0.05))
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(labelsize=14)
    mean_train = np.mean(np.array(pred)-np.array(solu))
    std_train = np.std(np.array(pred)-np.array(solu))
    # ax.set_xlim([-abs(x).max(), abs(x).max()])
    ax.set_xlim([-0.5, 0.5])
    plt.xlabel(r'$\log_{10} (E_{\rm pred})-\log_{10} (E_{\rm real})$',
               fontsize=14)
    plt.ylabel(r'N', fontsize=14)
    plt.text(abs(x).max()/3, y.max()-10,
             r'$\rm Mean = {0:.4f}$'.format(mean_train), fontsize=14)
    plt.text(abs(x).max()/3, y.max()-10-y.max()/10,
             r'$\rm Std = {0:.4f}$'.format(std_train), fontsize=14)
    plt.savefig(PATH_fig+'HistTest_'+name_prop+'.pdf')
    plt.show()

    # =============================================================================


# USE TENSOR BOARD TO VISUALIZE NETWORK ETC.
# Useful?

if tensor_board:

    tensor_x = torch.load('data/tensor_x_dens.pt')
    tensor_y = np.log10(torch.load('data/tensor_y_dens.pt'))

    # create your datset
    my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)

    train_length = tensor_x.shape[0] - 100
    test_length = 100
    lengths = [train_length, test_length]
    train_dataset, test_dataset = torch.utils.data.random_split(my_dataset,
                                                                lengths)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   shuffle=True, batch_size=1)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset,shuffle=True)

    writer = SummaryWriter('./runs/DenseArray_Experiment_1')

    for i in range(10):
        # get some random training images
        dataiter = iter(train_dataloader)
        images, labels = dataiter.next()

        # create grid of images
        img_grid = torchvision.utils.make_grid(images)

        # show images
        # plt.imshow(img_grid[0,:,:])

        # write to tensorboard
        writer.add_image('Array_images', img_grid)

    # helper function
    def select_n_random(data, labels, n=10):
        """Select n random datapoints and corresponding labels."""
        assert len(data) == len(labels)

        perm = torch.randperm(len(data))
        return data[perm][:n], labels[perm][:n]

    # select random images and their target indices
    images, labels = select_n_random(tensor_x, tensor_y)

    # get the class labels for each image
    # class_labels = [tensor_y[lab] for lab in labels]

    # log embeddings
    # features = images.view(-1, 13 * 25)
    # writer.add_embedding(features,
    #                     label_img=images.unsqueeze(1))

    writer.add_graph(net, images)
    writer.close()

    # tensorboard --logdir=runs/ --host=localhost
    # https://localhost:6006
    # http://localhost:8088
