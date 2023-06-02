import argparse
import os
import time

import h5py
import numpy as np
import torch
import torch.functional as F
from torch.utils import data
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
import ResNet18


def train(net, use_cuda, train_data_loader, optim, criterion):
    net.train()
    for _, data in enumerate(BackgroundGenerator(train_data_loader)):
        img, label = data
        if use_cuda:
            img = img.cuda()
            label = label.cuda()
        optim.zero_grad()
        y_pred = net(img)
        loss = criterion(y_pred, label)
        loss.backward()
        optim.step()
    return loss.item()


def test(model, use_cuda, test_data_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for img, label in test_data_loader:
            if use_cuda:
                img = img.cuda()
                label = label.cuda()
            y_pred = model(img)
            test_loss += criterion(y_pred, label).item()  # sum up batch loss
            # get the index of the max log-probability
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_data_loader.dataset)
    return (test_loss, correct)


def loaddata(dataset, emdtype, rind):
    matfiles = ["4_8", "8_10", "10_13", "13_30"]
    n_fold = 10
    if dataset == 1:
        n_chn = 19
        n_AD_test = 7
        n_CR_test = 14
    else:
        n_chn = 21
        n_AD_test = 12
        n_CR_test = 28

    data_AD_train_enh_container = np.zeros([500, n_chn, n_chn, 4])
    data_CR_train_enh_container = np.zeros([500, n_chn, n_chn, 4])
    data_AD_train_org_container = np.zeros([10, n_chn, n_chn, 4])
    data_CR_train_org_container = np.zeros([10, n_chn, n_chn, 4])
    data_AD_test_org_container = np.zeros([n_AD_test, n_chn, n_chn, 4])
    data_CR_test_org_container = np.zeros([n_CR_test, n_chn, n_chn, 4])
    targ_AD_train_enh_container = np.zeros([500, 4])
    targ_CR_train_enh_container = np.zeros([500, 4])
    targ_AD_train_org_container = np.zeros([10, 4])
    targ_CR_train_org_container = np.zeros([10, 4])
    targ_AD_test_org_container = np.zeros([n_AD_test, 4])
    targ_CR_test_org_container = np.zeros([n_CR_test, 4])

    for i in range(len(matfiles)):
        container = h5py.File("EnhData/Dataset_"+str(dataset)+"/CoMatrix/"+emdtype+"/comatrix_"+matfiles[i]+"_enh_rind_"+str(rind)+".mat", "r")
        data_AD_train_enh_container[:, :, :, i] = container["train_AD_enh"]
        data_CR_train_enh_container[:, :, :, i] = container["train_CR_enh"]
        data_AD_train_org_container[:, :, :, i] = container["train_AD"]
        data_CR_train_org_container[:, :, :, i] = container["train_CR"]
        data_AD_test_org_container[:, :, :, i] = container["test_AD"]
        data_CR_test_org_container[:, :, :, i] = container["test_CR"]

        targ_AD_train_enh_container[:, i] = container["lbl_train_AD_enh"]
        targ_CR_train_enh_container[:, i] = container["lbl_train_CR_enh"]
        targ_AD_train_org_container[:, i] = container["lbl_train_AD"]
        targ_CR_train_org_container[:, i] = container["lbl_train_CR"]
        targ_AD_test_org_container[:, i] = container["lbl_test_AD"]
        targ_CR_test_org_container[:, i] = container["lbl_test_CR"]

    x1e_train = torch.Tensor(data_AD_train_enh_container.transpose([0, 3, 1, 2]))
    x2e_train = torch.Tensor(data_CR_train_enh_container.transpose([0, 3, 1, 2]))
    x1o_train = torch.Tensor(data_AD_train_org_container.transpose([0, 3, 1, 2]))
    x2o_train = torch.Tensor(data_CR_train_org_container.transpose([0, 3, 1, 2]))
    x1o_test = torch.Tensor(data_AD_test_org_container.transpose([0, 3, 1, 2]))
    x2o_test = torch.Tensor(data_CR_test_org_container.transpose([0, 3, 1, 2]))

    y1e_train = torch.Tensor(targ_AD_train_enh_container[:, 0]-1.0).long()
    y2e_train = torch.Tensor(targ_CR_train_enh_container[:, 0]-1.0).long()
    y1o_train = torch.Tensor(targ_AD_train_org_container[:, 0]-1.0).long()
    y2o_train = torch.Tensor(targ_CR_train_org_container[:, 0]-1.0).long()
    y1o_test = torch.Tensor(targ_AD_test_org_container[:, 0]-1.0).long()
    y2o_test = torch.Tensor(targ_CR_test_org_container[:, 0]-1.0).long()

    return (x1e_train, x2e_train,x1o_train,x2o_train,x1o_test,x2o_test,y1e_train,y2e_train,y1o_train,y2o_train,y1o_test,y2o_test)

if __name__ == '__main__':
    lr=1e-4
    batch_size=0
    epochs=100
    torch.backends.cudnn.benchmark = True
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    criterion = torch.nn.CrossEntropyLoss()
    N_CHANNEL=[19,21]
    for dataset in [1, 2]:
        for emdtype in ["MEMD", "SEMD", "CEMD"]:
            for rind in [1,2,3,4,5,6,7,8,9,10]:
                # BrainNet_lr4 is  lr3
                # BrainNet_lr4_2 is lr4
                path="Model/Dataset_"+str(dataset)+"/"+emdtype+"/BrainNet_lr4_2"
                try:
                    os.makedirs(path)
                except:
                    None
                x1e_train, x2e_train,x1o_train,x2o_train,x1o_test,x2o_test,y1e_train,y2e_train,y1o_train,y2o_train,y1o_test,y2o_test=loaddata(dataset, emdtype, rind)
                train_loss_set = []
                test_loss_set = []
                correct_set = []
                for trial in [0,5,10,15,20,25,30,35,40,45,50,100,150,200,250,300,350,400,450,500]:
                    # if os.path.isfile(path+"/info_fold"+str(rind)+'_nenh'+str(trial)+'.mat'):
                    #     continue
                    if trial<=50:
                        batch_size=50
                    else:
                        batch_size=50
                    torch.backends.cudnn.benchmark = True
                    np.random.seed(1)
                    torch.manual_seed(1)
                    torch.cuda.manual_seed(1)
                    torch.cuda.manual_seed_all(1)
                    if dataset==1:
                        net = ResNet18.BasicBRNN(4, 2, 19)
                    else:
                        net = ResNet18.BasicBRNN(4, 2, 21)
                    use_cuda = torch.cuda.is_available()
                    if use_cuda:
                        net = net.cuda()

                    optim = torch.optim.Adam(net.parameters(), lr=lr)
                    if trial == 0:
                        train_container = TensorDataset(torch.cat([x1o_train, x2o_train], 0),
                            torch.cat([y1o_train, y2o_train], 0))
                    else:
                        train_container = TensorDataset(
                            torch.cat([x1o_train, x2o_train, x1e_train[0:trial, :, :, :], x2e_train[0:trial, :, :, :]], 0),
                            torch.cat([y1o_train, y2o_train, y1e_train[0:trial],          y2e_train[0:trial]], 0))
                    train_data_loader = DataLoader(
                        train_container, batch_size=batch_size, shuffle=True)
                    test_container = TensorDataset(
                        torch.cat([x1o_test,x2o_test], 0),
                        torch.cat([y1o_test,y2o_test], 0))
                    test_data_loader = DataLoader(test_container)

                    for epoch in tqdm(range(epochs), desc="Processing: "+"dataset_"+str(dataset)+" "\
                        +' Rind'+str(rind)+emdtype+str(trial)):
                        train_loss = train(
                            net, use_cuda, train_data_loader, optim, criterion)
                        test_loss, correct = test(
                            net, use_cuda, test_data_loader, criterion)
                        train_loss_set = train_loss_set+[train_loss]
                        test_loss_set = test_loss_set+[test_loss]
                        correct_set = correct_set+[correct]
                        # if ((epoch+1)%10)==0:
                        # torch.save(
                        #     net.state_dict(), path+"/model_"+str(trial)+"/epoch_"+str(epoch)+".pkl")
                        time.sleep(0.005)
                        pass

                    # for epoch in tqdm(range(opt.epochs)):
                    #     net.load_state_dict(torch.load(
                    #             opt.path[0]+"/model_"+str(trial)+"/epoch_"+str(epoch)+".pkl"))
                    #     test_loss, correct=test(net,use_cuda, test_data_loader, criterion)
                    #     test_loss_set=test_loss_set+[test_loss]
                    #     correct_set=correct_set+[correct]
                    #     time.sleep(0.01)
                    #     pass

                    sio.savemat(path+"/info_fold"+str(rind)+'_nenh'+str(trial)+'.mat', {
                        "train_loss": train_loss_set, "test_loss": test_loss_set, "correct": correct_set})
