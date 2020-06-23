#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:33:58 2020

@author: guepin
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
import torch_geometric.nn as nn
import torch.nn as Basenn
import torch_geometric.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import hdf5fileinout as hdf5io
import ComputePeak2PeakOnHDF5 as ComputeP2P
import glob
import random
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Set the random set for repeatability
manualSeed = 1
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

# =============================================================================

# CHOOSE PARAMETERS

# Choose progenitor: 'Proton'
progenitor = 'Proton'

# Choose zenith angle: 81.3
ZenVal = '81.3'

# Choose signal: 'efield', 'voltage', 'filteredvoltage', 'filteredvoltagenoise'
trace = 'efield'

# Choose labels to be predicted by the network: 'energy' or 'energy_azimuth'
net_labels = 'energy'

# Choose number of epochs for training
n_epochs = 500

# Choose learning rate, scheduler 'cst' or 'dec'
learn_rate = 1e-4
lr_scheduler = 'cst'

# Choose weight decay
wd = 0.0

# Choose batch size: 1 fully stochastic gradient descent
batchsize = 1

# Path for data bank (ZHAireS simulations)
PATH_data = '/Users/guepin/Documents/GRAND/TheGP300Outbox/'

# Name with chosen properties for saving information and figures
name_prop = trace+'_int_'+net_labels+'_'+progenitor+'_zen'+ZenVal+'_lr'\
    + str(learn_rate)+'_'+lr_scheduler+'_wd'+str(wd)+'_drop'\
    + '_bs'+str(batchsize)+'_nepoch'+str(n_epochs)

# Path to save trained network properties
PATH = './data/net_'+name_prop+'.pth'

# Path to save figures
PATH_fig = '/Users/guepin/Documents/NotesLatex/GRAND_reco/Figures_GP300_GNN'

FILE_data = glob.glob(PATH_data+'*'+progenitor+'*'+ZenVal+'*')
print('Number of files = %i' % (len(FILE_data)))

# All antenna positions
ant_pos_all = np.loadtxt('data/GP300propsedLayout.dat', usecols=(2, 3, 4))

# =============================================================================


class StarShapeDataset(InMemoryDataset):
    """Data set definition."""

    def __init__(self, root, transform=None, pre_transform=None):
        super(StarShapeDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """Define raw names."""
        return []

    @property
    def processed_file_names(self):
        """Define processed names."""
        return ['data.pt']

    def download(self):
        """Download to self.raw_dir."""
        pass

    def process(self):
        """Read data into huge Data list."""
        data_list = []

        for i in range(len(FILE_data)):

            inputfilename = glob.glob(FILE_data[i] + '/*' + progenitor + '*'
                                      + ZenVal + '*.hdf5')[0]
            RunInfo = hdf5io.GetRunInfo(inputfilename)
            EventName = hdf5io.GetEventName(RunInfo, 0)
            AntennaInfo = hdf5io.GetAntennaInfo(inputfilename, EventName)
            nantennas = hdf5io.GetNumberOfAntennas(AntennaInfo)
            energy = RunInfo['Energy'][0]

            # COMPUTE TOTAL PEAK TO PEAK AMPLITUDE
            p2pE = ComputeP2P.get_p2p_hdf5(inputfilename, antennamax='All',
                                           antennamin=0, usetrace='efield')
            # FILL TOTAL P2P FOR ALL 'TRIGGERED' ANTENNAS
            p2p_tot = np.zeros(len(ant_pos_all))
            for i_ant in range(nantennas):
                num_ant = np.int(AntennaInfo[i_ant][0][1:])
                p2p_tot[num_ant] = p2pE[3, i_ant]

            data = Data(x=torch.tensor(np.transpose(np.array([p2p_tot]))),
                        y=torch.tensor(np.array([np.log10(energy)])),
                        pos=torch.tensor(ant_pos_all))

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def accuracy_model(model, real_labels, pred_labels, maxlogdiff):
    """Test the accuracy of the energy reconstruction."""
    n_items = len(pred_labels)
    n_correct = torch.sum((torch.abs(real_labels - pred_labels) < maxlogdiff))
    result = (n_correct.item() * 100.0 / n_items)
    return result


class Net(torch.nn.Module):
    """Define Graph Neural Network."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.GraphConv(dataset.num_node_features, 6)
        self.conv2 = nn.GraphConv(6, 1)
        self.fc1 = Basenn.Linear(288, 288)
        self.fc2 = Basenn.Linear(288, 1)

    def forward(self, data):
        """Forward propagation."""
        x, edge_index = data.x.float(), data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = data.x.float()
        x = x.reshape(288)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # print(np.shape(x))

        return x


# =============================================================================

dataset = StarShapeDataset('./', pre_transform=T.KNNGraph(k=3))

model = Net()
dataset = dataset.shuffle()
# data = dataset[0]
# data.x = data.x.float()
# model(data)

max_P2P = torch.max(dataset.data.x).item()
dataset.data.x /= max_P2P

optimizer = torch.optim.Adam(model.parameters(),
                             lr=learn_rate,
                             weight_decay=wd)
crit = torch.nn.MSELoss()

nsim = len(dataset)
train_dataset = dataset[:nsim-50]
test_dataset = dataset[nsim-50:nsim]
train_loader = DataLoader(train_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1)

loss_cumul_arr = []

print("START TRAINING")
model.train()
for epoch in range(n_epochs):
    running_loss = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        label = data.y.float()
        loss = crit(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    loss_cumul_arr.append(running_loss)
    # print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, running_loss))

# Calculate the accuracy of training
solu = []
pred = []
with torch.no_grad():
    for data in train_loader:
        labels = data.y.float()
        outputs = model(data)
        for k in range(len(labels)):
            solu.append(labels[0].item())
            pred.append(outputs[0].item())

accuracy_train = accuracy_model(model, torch.Tensor(solu),
                                torch.Tensor(pred), 0.1)
print('Accuracy train: %0.2f %%' % (accuracy_train))

# =============================================================================

model.eval()

# Calculate the accuracy, test
solu_test = []
pred_test = []
with torch.no_grad():
    for data in test_loader:
        labels = data.y.float()
        outputs = model(data)
        for k in range(len(labels)):
            solu_test.append(labels[0].item())
            pred_test.append(outputs[0].item())

accuracy_test = accuracy_model(model, torch.Tensor(solu_test),
                               torch.Tensor(pred_test), 0.1)
print('Accuracy test: %0.2f %%' % (accuracy_test))

# =============================================================================
# FIGURES
# =============================================================================

fig = plt.figure()
ax = plt.gca()
plt.plot(loss_cumul_arr, linewidth=2)
plt.xlabel(r'Number of epochs', fontsize=16)
plt.ylabel(r'Cumulative loss', fontsize=16)
ax.set_xlim([0, n_epochs])
# ax.set_ylim([0, 50.])
ax.tick_params(labelsize=14)
plt.savefig(PATH_fig+'CumulLoss_'+name_prop+'.pdf')
plt.show()

# =============================================================================

fig = plt.figure()
ax = plt.gca()

y, x, _ = plt.hist(np.array(pred) - np.array(solu),
                   bins=np.arange(-0.5, 0.5 + 0.05, 0.05))

mean_train = np.mean(np.array(pred)-np.array(solu))
std_train = np.std(np.array(pred)-np.array(solu))

ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labelsize=14)
ax.set_xlim([-0.5, 0.5])
plt.xlabel(r'$\log_{10} (E_{\rm pred})-\log_{10} (E_{\rm real})$',
           fontsize=14)
plt.ylabel(r'$N$', fontsize=14)

plt.text(abs(x).max()/3, y.max()*9/10.,
         r'$\rm Mean = {0:.4f}$'.format(mean_train), fontsize=14)
plt.text(abs(x).max()/3, y.max()*9/10.-y.max()/10,
         r'$\rm Std = {0:.4f}$'.format(std_train), fontsize=14)
plt.text(abs(x).max()/3, y.max()*9/10.-y.max()*2/10,
         r'$\rm Accuracy = {0:.0f} \%$'.format(accuracy_train), fontsize=14)

plt.savefig(PATH_fig+'HistTrain_'+name_prop+'.pdf')
plt.show()

# =============================================================================

fig = plt.figure()
ax = plt.gca()

y, x, _ = plt.hist(np.array(pred_test) - np.array(solu_test),
                   bins=np.arange(-0.5, 0.5 + 0.05, 0.05))

mean_test = np.mean(np.array(pred_test)-np.array(solu_test))
std_test = np.std(np.array(pred_test)-np.array(solu_test))

ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labelsize=14)
ax.set_xlim([-0.5, 0.5])
plt.xlabel(r'$\log_{10} (E_{\rm pred})-\log_{10} (E_{\rm real})$',
           fontsize=14)
plt.ylabel(r'N', fontsize=14)

plt.text(abs(x).max()/3, y.max()*9/10.,
         r'$\rm Mean = {0:.4f}$'.format(mean_test), fontsize=14)
plt.text(abs(x).max()/3, y.max()*9/10.-y.max()/10,
         r'$\rm Std = {0:.4f}$'.format(std_test), fontsize=14)
plt.text(abs(x).max()/3, y.max()*9/10.-y.max()*2/10,
         r'$\rm Accuracy = {0:.0f} \%$'.format(accuracy_test), fontsize=14)

plt.savefig(PATH_fig+'HistTest_'+name_prop+'.pdf')
plt.show()

# =============================================================================

fig = plt.figure()
ax = plt.gca()

plt.plot(solu_test, linestyle='', marker='o')
plt.plot(pred_test, linestyle='', marker='x')

ax.tick_params(labelsize=14)

# ax.set_xlim([-10000,10000])
# ax.set_ylim([-10000,10000])
# ax.axis('equal')
plt.show()
