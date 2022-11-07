#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRAND - Study energy reconstruction with Graph Neural Network.

Created on Thu Jun  4 17:33:58 2020

@author: guepin
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DenseDataLoader
from torch_geometric.data import InMemoryDataset
import torch_geometric.nn as nn
import torch_geometric.utils as ut
import torch.nn as Basenn
import torch_geometric.transforms as Trans
import numpy as np
from matplotlib import cm
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
ZenVal = '74.8'

# Choose signal: 'efield', 'voltage', 'filteredvoltage', 'filteredvoltagenoise'
trace = 'efield'

# Choose labels to be predicted by the network: 'energy' or 'energy_azimuth'
net_labels = 'energy'

# Choose number of epochs for training
n_epochs = 100

# Choose learning rate, scheduler 'cst' or 'dec'
learn_rate = 1e-4
lr_scheduler = 'cst'

# Choose weight decay
wd = 0.0

# Choose batch size: 1 fully stochastic gradient descent
batchsize = 1

# Accuracy required
maxlog = 0.1

# Name with chosen properties for saving information and figures
name_prop = trace+'_p2p_'+net_labels+'_'+progenitor+'_zen'+ZenVal+'_lr'\
    + str(learn_rate)+'_'+lr_scheduler+'_wd'+str(wd)\
    + '_bs'+str(batchsize)+'_nepoch'+str(n_epochs)+'_GNN_8'

# All antenna positions
ant_pos_all = np.loadtxt('data/GP300propsedLayout.dat', usecols=(2, 3, 4))
max_dist = np.max(ant_pos_all)

# Path for data bank (ZHAireS simulations)
PATH_data = '/Users/claireguepin/Projects/GRAND/GP300Outbox/'
# Path to save figures
PATH_fig = '/Users/claireguepin/Figures/GRAND/'
# Path to save trained network properties
PATH = './data/net_'+name_prop+'.pth'

FILE_data = glob.glob(PATH_data+'*'+progenitor+'*'+ZenVal+'*')
print('Number of files = %i' % (len(FILE_data)))
# =============================================================================


class p2pDataset(InMemoryDataset):
    """Data set definition."""

    def __init__(self, root, transform=None, pre_transform=None):
        super(p2pDataset, self).__init__(root, transform, pre_transform)
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
        """Read data into data list."""
        data_list = []

        for i in range(len(FILE_data)):

            inputfilename = glob.glob(FILE_data[i] + '/*' + progenitor + '*'
                                      + ZenVal + '*.hdf5')[0]
            RunInfo = hdf5io.GetRunInfo(inputfilename)
            EventName = hdf5io.GetEventName(RunInfo, 0)
            AntennaInfo = hdf5io.GetAntennaInfo(inputfilename, EventName)
            nantennas = hdf5io.GetNumberOfAntennas(AntennaInfo)
            energy = RunInfo['Energy'][0]

            # FILL Efield array FOR ALL 'TRIGGERED' ANTENNAS
            efield_arr = np.zeros((len(ant_pos_all), 1100))
            for i_ant in range(nantennas):
                AntennaID = hdf5io.GetAntennaID(AntennaInfo, i_ant)
                efield_loc = hdf5io.GetAntennaEfield(
                    inputfilename, EventName, AntennaID)
                num_ant = int(AntennaInfo[i_ant][0][1:])
                efield_arr[num_ant, :] = efield_loc[0:1100, 1]
                # efield_loc = hdf5io.GetAntennaFilteredVoltage(
                #     inputfilename, EventName, AntennaID)
                # num_ant = int(AntennaInfo[i_ant][0][1:])
                # efield_arr[num_ant, :] = efield_loc[0:1100, 1]

            data = Data(
                x=torch.tensor(efield_arr),
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


class GNN(torch.nn.Module):
    """Define graph convolution layers."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.prelus = torch.nn.ModuleList()

        self.convs.append(nn.DenseGCNConv(in_channels, out_channels))
        self.prelus.append(torch.nn.PReLU())

        # self.convs.append(nn.DenseGCNConv(in_channels, hidden_channels))
        # self.prelus.append(torch.nn.PReLU())
        # self.convs.append(nn.DenseGCNConv(hidden_channels, hidden_channels))
        # self.prelus.append(torch.nn.PReLU())
        # self.convs.append(nn.DenseGCNConv(hidden_channels, out_channels))
        # self.prelus.append(torch.nn.PReLU())

    def forward(self, x, adj, mask=None):
        """Forward propagation."""
        batch_size, num_nodes, in_channels = x.size()

        for step in range(len(self.convs)):
            x = self.prelus[step](self.convs[step](x, adj, mask))

        return x


class Net(torch.nn.Module):
    """Define Graph Neural Network using graph convolution layers."""

    def __init__(self):
        super(Net, self).__init__()
        # Check if num input, hidden and output channels are correctly defined
        # Check what parameter lin does. Unclear
        num_hid_cha = 12
        num_out_cha = 12

        num_inp_cha = dataset.num_node_features+3
        # num_hid_cha = 64
        num_nodes = int(np.ceil(0.1 * 288))  # 0.25/0.33
        # num_out_cha = 64
        self.conv1_pool = GNN(num_inp_cha, num_hid_cha, num_nodes)
        self.conv1_emb = GNN(num_inp_cha, num_hid_cha, num_out_cha)

        num_inp_cha = num_out_cha
        # num_hid_cha = 64
        num_nodes = int(np.ceil(0.1 * num_nodes))  # 0.25/0.33
        # num_out_cha = 64
        self.conv2_pool = GNN(num_inp_cha, num_hid_cha, num_nodes)
        self.conv2_emb = GNN(num_inp_cha, num_hid_cha, num_out_cha, lin=False)

        num_inp_cha = num_out_cha
        num_hid_cha = 4
        num_out_cha = 4
        self.gnn3_emb = GNN(num_inp_cha, num_hid_cha, num_out_cha, lin=False)

        self.fc1 = Basenn.Linear(12, 4)  # 36/192/216/384/1024/2048
        self.fc2 = Basenn.Linear(4, 1)
        self.prelufc1 = torch.nn.PReLU()

        # self.fc1 = Basenn.Linear(316800, 20)  # 192/216/384/1024/2048
        # self.fc2 = Basenn.Linear(20, 1)
        # self.prelufc1 = torch.nn.PReLU()

    def forward(self, data, mask=None):
        """Forward propagation."""
        # x, adj = data.x.float(), data.adj
        x = torch.cat((data.x.float(), data.pos.float()/max_dist), 2)
        # print(np.shape(x))
        adj = data.adj
        s = self.conv1_pool(x, adj, mask)
        x = self.conv1_emb(x, adj, mask)
        # print(np.shape(x))
        x, adj, l1, e1 = nn.dense_diff_pool(x, adj, s, mask)
        # print(np.shape(x))
        s = self.conv2_pool(x, adj)
        x = self.conv2_emb(x, adj)
        # print(np.shape(x))
        x, adj, l2, e2 = nn.dense_diff_pool(x, adj, s)
        # print(np.shape(x))
        x = self.gnn3_emb(x, adj)
        # print(np.shape(x))
        x = torch.reshape(x, (-1,))
        # print(np.shape(x))
        x = self.prelufc1(self.fc1(x))
        # print(np.shape(x))
        x = self.fc2(x)
        return x


def count_parameters(model):
    """Count number of parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================

dataset = p2pDataset('./', pre_transform=Trans.Compose([
    Trans.KNNGraph(k=6, loop=True, force_undirected=True), Trans.ToDense()]))

# dataset = p2pDataset('./')
dataset = dataset.shuffle()
max_P2P = torch.max(dataset.data.x).item()
# print("\nMaximum peak to peak value used to normalize data: %.2e" % max_P2P)
dataset.data.x /= max_P2P

model = Net()
print("Number of parameters: %i" % count_parameters(model))

nsim = len(dataset)
len_train = int(3.*nsim/4.)
train_dataset = dataset[:len_train]
test_dataset = dataset[len_train+1:nsim]
# train_dataset = dataset[:nsim-20]
# test_dataset = dataset[nsim-20:nsim]
train_loader = DenseDataLoader(train_dataset, batch_size=batchsize)
test_loader = DenseDataLoader(test_dataset, batch_size=batchsize)

optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate,
                             weight_decay=wd)
crit = torch.nn.MSELoss()

# data = dataset[0]
# model(data)

# for data in train_loader:
#     print(data.y)
#     model(data)
#     break

# =============================================================================
# VISUALIZE ARRAY

# plt.figure()
# ax = plt.gca()

# plt.scatter(np.array(dataset[0].pos[:, 0])/1e3,
#             np.array(dataset[0].pos[:, 1])/1e3,
#             30,
#             c=np.array(dataset[0].x)/np.max(np.array(dataset[0].x)),
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

loss_cumul_arr = []

print("\nSTART TRAINING")
model.train()
for epoch in range(n_epochs):
    # print("epoch: %i" % epoch)
    running_loss = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        label = data.y[0].float()
        loss = crit(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    loss_cumul_arr.append(running_loss)
    # print("cumul loss: %.2f" % running_loss)
    print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, running_loss))

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
                                torch.Tensor(pred), maxlog)
rms_train = np.std(np.array(pred)-np.array(solu))
print('Accuracy train: %0.2f %%' % (accuracy_train))
print('RMS log10 train: %0.3f' % (rms_train))

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
                               torch.Tensor(pred_test), maxlog)
rms_test = np.std(np.array(pred_test)-np.array(solu_test))
print('Accuracy test: %0.2f %%' % (accuracy_test))
print('RMS log10 test: %0.3f' % (rms_test))

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
# plt.text(abs(x).max()/3, y.max()*9/10.-y.max()*2/10,
#           r'$\rm Accuracy = {0:.0f} \%$'.format(accuracy_train), fontsize=14)

plt.savefig(PATH_fig+'HistTrain_log10_'+name_prop+'.pdf')
plt.show()

fig = plt.figure()
ax = plt.gca()

# DE_E = (10**np.array(pred) - 10**np.array(solu))/10**np.array(solu)
DE_E = 10**np.array(pred)/10**np.array(solu)

# y, x, _ = plt.hist(DE_E, bins=np.arange(-0.5, 0.5 + 0.05, 0.05))
y, x, _ = plt.hist(DE_E, bins=np.arange(0., 2. + 0.05, 0.05))
mean_train = np.mean(DE_E)
std_train = np.std(DE_E)

ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labelsize=14)
# ax.set_xlim([-0.5, 0.5])
ax.set_xlim([0., 2.])
plt.xlabel(r'$E_{\rm pred}/E_{\rm real}$', fontsize=14)
plt.ylabel(r'$N$', fontsize=14)

plt.text(abs(x).max()/3, y.max()*9/10.,
         r'$\rm Mean = {0:.4f}$'.format(mean_train), fontsize=14)
plt.text(abs(x).max()/3, y.max()*9/10.-y.max()/10,
         r'$\rm Std = {0:.4f}$'.format(std_train), fontsize=14)
# plt.text(abs(x).max()/3, y.max()*9/10.-y.max()*2/10,
#           r'$\rm Accuracy = {0:.0f} \%$'.format(accuracy_train), fontsize=14)
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
# plt.text(abs(x).max()/3, y.max()*9/10.-y.max()*2/10,
#           r'$\rm Accuracy = {0:.0f} \%$'.format(accuracy_test), fontsize=14)

plt.savefig(PATH_fig+'HistTest_log10_'+name_prop+'.pdf')
plt.show()

fig = plt.figure()
ax = plt.gca()

DE_E = 10**np.array(pred_test)/10**np.array(solu_test)

# y, x, _ = plt.hist(DE_E, bins=np.arange(-0.5, 0.5 + 0.05, 0.05))
y, x, _ = plt.hist(DE_E, bins=np.arange(0., 2. + 0.05, 0.05))
mean_train = np.mean(DE_E)
std_train = np.std(DE_E)

ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labelsize=14)
# ax.set_xlim([-0.5, 0.5])
ax.set_xlim([0., 2.])
plt.xlabel(r'$E_{\rm pred}/E_{\rm real}$', fontsize=14)
plt.ylabel(r'$N$', fontsize=14)

plt.text(abs(x).max()/3, y.max()*9/10.,
         r'$\rm Mean = {0:.4f}$'.format(mean_train), fontsize=14)
plt.text(abs(x).max()/3, y.max()*9/10.-y.max()/10,
         r'$\rm Std = {0:.4f}$'.format(std_train), fontsize=14)
# plt.text(abs(x).max()/3, y.max()*9/10.-y.max()*2/10,
#           r'$\rm Accuracy = {0:.0f} \%$'.format(accuracy_train), fontsize=14)
plt.savefig(PATH_fig+'HistTest_'+name_prop+'.pdf')
plt.show()

# =============================================================================

fig = plt.figure()
ax = plt.gca()
plt.subplots_adjust(left=0.13)

plt.plot(solu, linestyle='', marker='o', label='Real')
plt.plot(pred, linestyle='', marker='x', label='Predicted, train')

ax.tick_params(labelsize=14)
plt.xlabel(r'Simulation number', fontsize=14)
plt.ylabel(r'$\log_{10} (E)$', fontsize=14)
plt.legend(frameon=True, fontsize=14)

plt.savefig(PATH_fig+'DataTrain_'+name_prop+'.pdf')

# ax.set_xlim([-10000,10000])
# ax.set_ylim([-10000,10000])
# ax.axis('equal')
plt.show()

# =============================================================================

fig = plt.figure()
ax = plt.gca()
plt.subplots_adjust(left=0.13)

plt.plot(solu_test, linestyle='', marker='o', label='Real')
plt.plot(pred_test, linestyle='', marker='x', label='Predicted, test')

ax.tick_params(labelsize=14)
plt.xlabel(r'Simulation number', fontsize=14)
plt.ylabel(r'$\log_{10} (E)$', fontsize=14)
plt.legend(frameon=True, fontsize=14)

plt.savefig(PATH_fig+'DataTest_'+name_prop+'.pdf')

# ax.set_xlim([-10000,10000])
# ax.set_ylim([-10000,10000])
# ax.axis('equal')
plt.show()
