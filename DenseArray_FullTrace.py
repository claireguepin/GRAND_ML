#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:37:04 2020.

@author: guepin
"""

# =============================================================================
# GRAND MACHINE LEARNING FOR RECONSTRUCTION
# Goal: study energy reconstruction with convolutional neural network
# ADD FULL TRACE TREATMENT
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

# Create data set for one progenitor, one zenith angle
create_dataset = 0

# Deep learning training and test
train_model = 1
test_model = 1

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
n_epochs = 500

# Choose learning rate, scheduler 'cst' or 'dec'
learn_rate = 1e-4
lr_scheduler = 'cst'

# Choose weight decay
wd = 0.01

# Choose batch size: 1 fully stochastic gradient descent
batchsize = 1

# Path for data bank (ZHAireS simulations)
# Outbox, OutboxVoltage, OutboxSignalProc
PATH_data = '/Users/guepin/Documents/GRAND/OutboxSignalProc/'

# Name with chosen properties for saving information and figures
name_prop = 'full_trace_'+net_labels+'_'+progenitor+'_zen'+ZenVal+'_'\
    + trace+'_lr'+str(learn_rate)+'_'+lr_scheduler+'_wd'+str(wd)+'_bs'\
    + str(batchsize)+'_nepoch'+str(n_epochs)

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
        # For time traces
        self.conv3d_1 = nn.Conv3d(4, 4, (21, 1, 1), stride=(5, 1, 1))
        self.pool3d_1 = nn.MaxPool3d((2, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 1, (21, 1, 1), stride=(1, 1, 1))
        self.pool3d_2 = nn.MaxPool3d((2, 1, 1))
        # For spatial properties
        self.conv1 = nn.Conv2d(4, 4, 4)
        self.conv2 = nn.Conv2d(4, 2, 2)
        # Default value of stride is kernel_size
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(2*2*5, 2*2*5)
        if net_labels == 'energy':
            self.fc2 = nn.Linear(2*2*5, 1)
        elif net_labels == 'energy_azimuth':
            self.fc2 = nn.Linear(2*2*5, 2)
        else:
            print('ERROR: CHOOSE LABELS!')

    def forward(self, x):
        """Forward propagation."""
        # Time traces properties
        x = self.pool3d_1(F.relu(self.conv3d_1(x)))
        x = self.pool3d_2(F.relu(self.conv3d_2(x)))
        x = torch.reshape(x, (1, 4, 13, 25))
        # Spatial properties
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 2*2*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()

# tensor_x = torch.Tensor(np.zeros(shape=(1, 4, 298, 13, 25)))
# tensor_x = torch.Tensor(np.zeros(shape=(1, 1, 13, 25)))
# net(tensor_x)

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

# CREATE DATASET for ML

# Specific data set: all energies for one zenith angle and one progenitor
if create_dataset:

    traces = []
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
            EventName = hdf5io.GetEventName(RunInfo, 0)
            AntennaInfo = hdf5io.GetAntennaInfo(inputfilename, EventName)
            nantennas = hdf5io.GetNumberOfAntennas(AntennaInfo)
            Zenith = 180.-hdf5io.GetEventZenith(RunInfo, 0)
            Azimuth = 360.-hdf5io.GetEventAzimuth(RunInfo, 0)

            filtvoltnoise_tab = np.zeros(shape=(4, 298, 13, 25))
            for l in range(13):
                for c in range(25):
                    index = c * 13 + (12-l)
                    filtvoltnoise_tab[:, :, l, c] \
                        = hdf5io.GetAntennaFilteredVoltageNoise(
                            inputfilename, EventName, index).T

            traces.append(filtvoltnoise_tab)

            if net_labels == 'energy':
                params.append(np.array([RunInfo['Energy'][0]]))
            elif net_labels == 'energy_azimuth':
                params.append(np.array([RunInfo['Energy'][0], Azimuth]))
            else:
                print('ERROR: CHOOSE LABELS!')

    # Transform to torch tensors
    tensor_x = torch.Tensor(traces)
    tensor_y = torch.Tensor(params)

    torch.save(tensor_x, 'data/tensor_x_tra_'+net_labels+'_'+progenitor
               + '_zen'+ZenVal+'_'+trace+'.pt')
    torch.save(tensor_y, 'data/tensor_y_tra_'+net_labels+'_'+progenitor
               + '_zen'+ZenVal+'_'+trace+'.pt')

# =============================================================================

# TRAIN MODEL

if train_model:

    time_ini = time.time()

    tensor_x = torch.load('data/tensor_x_tra_'+net_labels+'_'
                          + progenitor+'_zen'+ZenVal+'_'+trace+'.pt') / P2PMax
    tensor_y = torch.load('data/tensor_y_tra_'+net_labels+'_'
                          + progenitor+'_zen'+ZenVal+'_'+trace+'.pt')

    tensor_y[:, 0] = np.log10(tensor_y[:, 0])
    if len(tensor_y[0, :]) > 1:
        tensor_y[:, 1] /= 360.

    # Create dataset
    my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)

    test_length = 100
    train_length = tensor_x.shape[0]-test_length
    lengths = [train_length, test_length]
    train_dataset, test_dataset = torch.utils.data.random_split(my_dataset,
                                                                lengths)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                                   batch_size=batchsize)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True)

    # SAVE TRAIN AND TEST DATALOADERS
    torch.save(train_dataloader, 'data/train_dataloader_'+name_prop+'.pt')
    torch.save(test_dataloader, 'data/test_dataloader_'+name_prop+'.pt')

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
    # plt.savefig(PATH_fig+'CumulLoss_'+name_prop+'.pdf')
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
    # plt.savefig(PATH_fig+'Loss_'+name_prop+'.pdf')
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
    # plt.savefig(PATH_fig+'HistTrain_'+name_prop+'.pdf')
    plt.show()

# =============================================================================

# TEST MODEL

if test_model:

    net.load_state_dict(torch.load(PATH))
    net.eval()

    test_dataloader = torch.load('data/test_dataloader_'+name_prop+'.pt')

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
    # plt.savefig(PATH_fig+'HistTest_'+name_prop+'.pdf')
    plt.show()

    # =============================================================================
