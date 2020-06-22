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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import hdf5fileinout as hdf5io
import ComputePeak2PeakOnHDF5 as ComputeP2P
import glob
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# =============================================================================
def GetUVW(pos, cx, cy, cz, zen, az):
    """Rotate in shower plane."""
    relpos = pos-np.array([cx, cy, cz])
    inc = 152.95*np.pi/180.
    phigeo = 0.*np.pi/180.

    # from oliviers script including phigeo
    B = np.array([np.cos(phigeo)*np.sin(inc),
                  np.sin(phigeo)*np.sin(inc),
                  np.cos(inc)])
    B = B/np.linalg.norm(B)
    v = np.array([np.cos(az)*np.sin(zen),
                  np.sin(az)*np.sin(zen),
                  np.cos(zen)])  # or *-1: change the direction
    v = v/np.linalg.norm(v)
    vxB = np.cross(v, B)
    vxB = vxB/np.linalg.norm(vxB)
    vxvxB = np.cross(v, vxB)
    vxvxB = vxvxB/np.linalg.norm(vxvxB)
    return np.array([np.inner(vxB, relpos),
                     np.inner(vxvxB, relpos),
                     np.inner(v, relpos)]).T


# =============================================================================

PATH_data = '/Users/guepin/Documents/GRAND/UHECR_Xmax/CR-Sim/CR190/'
FILE_data = glob.glob(PATH_data+'*')

# p2p_list = []
# ene_list = []
# pos_list = []
# dat_list = []

# # for i in range(len(FILE_data)):
# for i in range(1):
#     file_ant = FILE_data[i]+'/antpos.dat'
#     pos_ant = np.loadtxt(file_ant)

#     file_info = glob.glob(FILE_data[i]+'/*inp')[0]
#     infos = np.loadtxt(file_info, dtype='str', max_rows=5, usecols=(0, 1))

#     primary = infos[1, 1]
#     energy = infos[2, 1]
#     zenith = infos[3, 1]
#     azimuth = infos[4, 1]

#     data_ene = float(energy)
#     data_zen = float(zenith)*np.pi/180.
#     data_azi = float(azimuth)*np.pi/180.

#     p2p_total = np.zeros(len(pos_ant))
#     for j in range(len(pos_ant)):
#         trace = np.loadtxt(FILE_data[i]+'/a'+str(j)+'.trace')
#         amplitude = np.sqrt(trace[:, 1]**2.+trace[:, 2]**2.+trace[:, 3]**2.)
#         p2p_total[j] = max(amplitude)-min(amplitude)

#     # rotates antennas in vxB-vxvxB
#     pos_ant_UVW = GetUVW(pos_ant, 0., 0., 0., data_zen, data_azi)

#     p2p_list.append(np.array([p2p_total]))
#     ene_list.append(np.array([data_ene]))
#     pos_list.append(np.array([pos_ant_UVW]))

#     data = Data(x=torch.tensor(p2p_total),
#                 y=torch.tensor([data_ene]),
#                 pos=torch.tensor(pos_ant_UVW))
#     dat_list.append(data)

# data = Data(x=torch.tensor(p2p_list),
#             y=torch.tensor(ene_list),
#             pos=torch.tensor(pos_list),
#             pre_transform=T.KNNGraph(k=6))

# torch.save(torch.tensor(p2p_list), 'p2p.pt')
# torch.save(torch.tensor(ene_list), 'ene.pt')
# torch.save(torch.tensor(pos_list), 'pos.pt')

# torch.save(torch.tensor(dat_list), 'data_list.pt')

# data_list = torch.load('data.pt')


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
            file_ant = FILE_data[i]+'/antpos.dat'
            pos_ant = np.loadtxt(file_ant)

            file_info = glob.glob(FILE_data[i]+'/*inp')[0]
            infos = np.loadtxt(file_info, dtype='str',
                               max_rows=5,
                               usecols=(0, 1))

            primary = infos[1, 1]
            # primary_ind = np.array([primary == 'Proton',
            #                         primary == 'Iron'])
            primary_ind = np.array([primary == 'Iron'])
            # energy = infos[2, 1]
            zenith = infos[3, 1]
            azimuth = infos[4, 1]

            # data_ene = np.log10(float(energy))
            data_zen = float(zenith)*np.pi/180.
            data_azi = float(azimuth)*np.pi/180.

            p2p_total = np.zeros(len(pos_ant))
            for j in range(len(pos_ant)):
                trace = np.loadtxt(FILE_data[i]+'/a'+str(j)+'.trace')
                amplitude = np.sqrt(trace[:, 1]**2.
                                    + trace[:, 2]**2.
                                    + trace[:, 3]**2.)
                p2p_total[j] = max(amplitude)-min(amplitude)

            # rotates antennas in vxB-vxvxB
            pos_ant_UVW = GetUVW(pos_ant, 0., 0., 0., data_zen, data_azi)

            data = Data(x=torch.tensor(np.transpose(np.array([p2p_total]))),
                        y=torch.tensor(primary_ind),
                        pos=torch.tensor(pos_ant_UVW))

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


dataset = StarShapeDataset('./', pre_transform=T.KNNGraph(k=6))
# from torch_geometric.datasets import Planetoid
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
# dataset[0]

# =============================================================================

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.GraphConv(dataset.num_node_features, 6)
#         self.conv2 = nn.GraphConv(6, 1)
#         self.fc1 = Basenn.Linear(160, 160)
#         self.fc2 = Basenn.Linear(160, 1)

#     def forward(self, data):
#         x, edge_index = data.x.float(), data.edge_index
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = x.reshape(160)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         # print(np.shape(x))

#         return x


class Net(torch.nn.Module):
    """Define Graph Neural Network."""

    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.PointConv()
        # self.conv2 = nn.PointConv()
        self.conv1 = nn.GraphConv(dataset.num_node_features, 6)
        self.conv2 = nn.GraphConv(6, 1)
        # self.fc1 = Basenn.Linear(1120, 160)
        self.fc1 = Basenn.Linear(160, 160)
        self.fc2 = Basenn.Linear(160, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):
        """Forward propagation."""
        # x, edge_index, pos = data.x.float(), data.edge_index, data.pos.float()
        # x, edge_index = data.x.float(), data.edge_index
        # x = self.conv1(x, pos, edge_index)
        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = self.conv2(x, pos, edge_index)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = x.reshape(1120)
        x = data.x.float()
        x = x.reshape(160)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        # print(np.shape(x))

        return x


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net()
dataset = dataset.shuffle()
# data = dataset[0]
# data.x = data.x.float()
# model(data)

max_P2P = torch.max(dataset.data.x).item()
dataset.data.x /= max_P2P

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
crit = torch.nn.BCELoss()
# crit = torch.nn.CrossEntropyLoss()

nsim = len(dataset)
train_dataset = dataset[:nsim-20]
test_dataset = dataset[nsim-20:nsim]
train_loader = DataLoader(train_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1)

loss_cumul_arr = []

model.train()
for epoch in range(500):
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

sol = np.array(solu)
pre = np.array(pred)

accuracy = np.sum(np.abs(sol-pre) < 0.5)/len(solu)*100.
accuracy_p = np.sum(np.abs(sol[sol == 0]-pre[sol == 0]) < 0.5)\
    / len(sol[sol == 0])*100.
accuracy_i = np.sum(np.abs(sol[sol == 1]-pre[sol == 1]) < 0.5)\
    / len(sol[sol == 1])*100.
print('Train accuracy = %.2f' % accuracy)
print('Train accuracy proton = %.2f' % accuracy_p)
print('Train accuracy iron = %.2f' % accuracy_i)

# =============================================================================

model.eval()

# Calculate the accuracy, test
solu = []
pred = []
with torch.no_grad():
    for data in test_loader:
        labels = data.y.float()
        outputs = model(data)
        for k in range(len(labels)):
            solu.append(labels[0].item())
            pred.append(outputs[0].item())

sol = np.array(solu)
pre = np.array(pred)

accuracy = np.sum(np.abs(sol-pre) < 0.5)/len(solu)*100.
accuracy_p = np.sum(np.abs(sol[sol == 0]-pre[sol == 0]) < 0.5)\
    / len(sol[sol == 0])*100.
accuracy_i = np.sum(np.abs(sol[sol == 1]-pre[sol == 1]) < 0.5)\
    / len(sol[sol == 1])*100.
print('Test accuracy = %.2f' % accuracy)
print('Test accuracy proton = %.2f' % accuracy_p)
print('Test accuracy iron = %.2f' % accuracy_i)

# =============================================================================

fig = plt.figure()
ax = plt.gca()
plt.plot(loss_cumul_arr, linewidth=2)
plt.xlabel(r'Number of epochs', fontsize=16)
plt.ylabel(r'Cumulative loss', fontsize=16)
# ax.set_xlim([0, len(loss_cumul_arr)])
ax.set_ylim([0, 50.])
ax.tick_params(labelsize=14)
# plt.savefig(PATH_fig+'CumulLoss_'+name_prop+'.pdf')
plt.show()

# fig = plt.figure()
# ax = plt.gca()
# # y, x, _ = plt.hist(np.array(pred) - np.array(solu),
# #                    bins=np.arange(-0.5, 0.5 + 0.05, 0.05))
# y, x, _ = plt.hist(np.array(pred) - np.array(solu), bins=10)
# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.tick_params(labelsize=14)
# mean_train = np.mean(np.array(pred)-np.array(solu))
# std_train = np.std(np.array(pred)-np.array(solu))
# # ax.set_xlim([-abs(x).max(), abs(x).max()])
# # ax.set_xlim([-0.5, 0.5])
# # plt.xlabel(r'$\log_{10} (E_{\rm pred})-\log_{10} (E_{\rm real})$',
# #            fontsize=14)
# plt.ylabel(r'$N$', fontsize=14)
# # plt.text(abs(x).max()/3, y.max()-10,
# #          r'$\rm Mean = {0:.4f}$'.format(mean_train), fontsize=14)
# # plt.text(abs(x).max()/3, y.max()-10-y.max()/10,
# #          r'$\rm Std = {0:.4f}$'.format(std_train), fontsize=14)
# # plt.savefig(PATH_fig+'HistTrain_'+name_prop+'.pdf')
# plt.show()

# model.eval()
# _, pred = model(data).max(dim=1)
# correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
# acc = correct / data.test_mask.sum().item()
# print('Accuracy: {:.4f}'.format(acc))

# =============================================================================
# PLOT ARRAY

# =============================================================================
# Projection on the ground

# fig = plt.figure()
# ax = plt.gca()

# ax.scatter(pos_ant[:, 0]/1e3, pos_ant[:, 1]/1e3,
#            c=p2p_total, marker='o')

# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.tick_params(labelsize=14)

# # ax.set_xlim([-10, 10])
# # ax.set_ylim([-10, 10])
# plt.subplots_adjust(left=0.14)
# ax.set_xlabel(r"$X\,{\rm\,(km)}$", fontsize=14)
# ax.set_ylabel(r"$Y\,{\rm\,(km)}$", fontsize=14)
# ax.axis('equal')

# plt.show()

# =============================================================================
# Projection in the shower plane

# fig = plt.figure()
# ax = plt.gca()

# ax.scatter(pos_ant_UVW[:, 0]/1e3, pos_ant_UVW[:, 1]/1e3,
#             c=p2p_total, marker='o')

# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.tick_params(labelsize=14)

# # ax.set_xlim([-10, 10])
# # ax.set_ylim([-10, 10])
# plt.subplots_adjust(left=0.14)
# ax.set_xlabel(r"$X\,{\rm\,(km)}$", fontsize=14)
# ax.set_ylabel(r"$Y\,{\rm\,(km)}$", fontsize=14)
# ax.axis('equal')

# plt.show()
