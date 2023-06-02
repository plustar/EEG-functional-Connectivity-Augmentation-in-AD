import argparse
import os
import time

import h5py
import numpy as np
import torch
import torch.functional as F
from torch.utils import data
from tqdm import tqdm
from torch import nn
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio

class EEGNet(nn.Module):
    def __init__(self,
                 num_samples: int = 2560,
                 num_channels: int = 21,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout

        self.block1 = nn.Sequential(
            nn.ZeroPad2d([self.kernel_1//2-1,self.kernel_1//2,0,0]),
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.Conv2d(self.F1,
                      self.F1 * self.D, (self.num_channels, 1),
                      groups=self.F1,
                      bias=False), 
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(), 
            nn.AvgPool2d((1, 4), stride=4), 
            nn.Dropout(p=dropout),
            nn.ZeroPad2d([self.kernel_2//2-1,self.kernel_2//2,0,0]),
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3), 
            nn.ELU(), 
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout)
        )


        self.block2 = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.F1 * self.D*self.num_samples//32, num_classes, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x


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
    pred_y=[]
    test_y=[]
    with torch.no_grad():
        i=0
        for img, label in test_data_loader:
            if use_cuda:
                img = img.cuda()
                label = label.cuda()
            y_pred = model(img)
            test_loss += criterion(y_pred, label).item()  # sum up batch loss
            # get the index of the max log-probability
            pred = y_pred.argmax(dim=1, keepdim=True)
            pred_y.append(pred.item())
            test_y.append(label.item())
            # correct += pred.eq(label.view_as(pred)).sum().item()
    # print(img.shape)
    test_loss /= len(test_data_loader.dataset)
    return (test_loss, np.asarray(pred_y),np.asarray(test_y))


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

    container = h5py.File("EnhData/Dataset_"+str(dataset)+"/SplitData/"+emdtype+"/enhd_Data_"+str(rind)+".mat", "r")
    data_AD_train_enh_container = np.array(container["train_AD_enh"])
    data_CR_train_enh_container = np.array(container["train_CR_enh"])
    data_AD_train_org_container = np.array(container["train_AD"])
    data_CR_train_org_container = np.array(container["train_CR"])
    data_AD_test_org_container = np.array(container["test_AD"])
    data_CR_test_org_container = np.array(container["test_CR"])

    targ_AD_train_enh_container = np.zeros([500,1])
    targ_CR_train_enh_container = np.ones([500,1])
    targ_AD_train_org_container = np.zeros([10,1])
    targ_CR_train_org_container = np.ones([10,1])
    targ_AD_test_org_container = np.zeros([np.shape(data_AD_test_org_container)[2],1])
    targ_CR_test_org_container = np.ones([np.shape(data_CR_test_org_container)[2],1])

    x1e_train = torch.Tensor(data_AD_train_enh_container).unsqueeze(1).permute([3,1,2,0])
    x2e_train = torch.Tensor(data_CR_train_enh_container).unsqueeze(1).permute([3,1,2,0])
    x1o_train = torch.Tensor(data_AD_train_org_container).unsqueeze(1).permute([3,1,2,0])
    x2o_train = torch.Tensor(data_CR_train_org_container).unsqueeze(1).permute([3,1,2,0])
    x1o_test = torch.Tensor(data_AD_test_org_container).unsqueeze(1).permute([3,1,2,0])
    x2o_test = torch.Tensor(data_CR_test_org_container).unsqueeze(1).permute([3,1,2,0])

    y1e_train = torch.Tensor(targ_AD_train_enh_container[:, 0]).long()
    y2e_train = torch.Tensor(targ_CR_train_enh_container[:, 0]).long()
    y1o_train = torch.Tensor(targ_AD_train_org_container[:, 0]).long()
    y2o_train = torch.Tensor(targ_CR_train_org_container[:, 0]).long()
    y1o_test = torch.Tensor(targ_AD_test_org_container[:, 0]).long()
    y2o_test = torch.Tensor(targ_CR_test_org_container[:, 0]).long()

    return (x1e_train, x2e_train,x1o_train,x2o_train,x1o_test,x2o_test,y1e_train,y2e_train,y1o_train,y2o_train,y1o_test,y2o_test)

if __name__ == '__main__':
    # EEGNet lr=1e-3
    # EEGNet2 lr=1e-4
    lr=1e-4
    batch_size=0
    epochs=200
    torch.backends.cudnn.benchmark = True
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    criterion = torch.nn.CrossEntropyLoss()
    N_CHANNEL=[19,21]
    for dataset in [1]:
        for emdtype in ["MEMD"]:
            for rind in [1,2,3,4,5,6,7,8,9,10]:
                path="Model/Dataset_"+str(dataset)+"/"+emdtype+"/EEGNet_lr4_conf"
                try:
                    os.makedirs(path)
                except:
                    None
                x1e_train, x2e_train,x1o_train,x2o_train,x1o_test,x2o_test,y1e_train,y2e_train,y1o_train,y2o_train,y1o_test,y2o_test=loaddata(dataset, emdtype, rind)
                train_loss_set = []
                test_loss_set = []
                pred_label = []
                test_label = []
                for trial in [0,10]:
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
                        net = EEGNet(2560, 21)
                    else:
                        net = EEGNet(4000, 21)
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
                        torch.cat([y1o_test,y2o_test], 0)
                    test_data_loader = DataLoader(test_container)

                    for epoch in tqdm(range(epochs), desc="Processing: "+"dataset_"+str(dataset)+" "\
                        +' Rind'+str(rind)+emdtype+str(trial)):
                        train_loss = train(
                            net, use_cuda, train_data_loader, optim, criterion)
                        test_loss, pred_y, test_y = test(
                            net, use_cuda, test_data_loader, criterion)
                        train_loss_set = train_loss_set+[train_loss]
                        test_loss_set = test_loss_set+[test_loss]

                        pred_label = pred_label+[pred_y]
                        test_label = test_label+[test_y]
                        # if ((epoch+1)%10)==0:
                        # torch.save(
                        #     net.state_dict(), path+"/model_"+str(trial)+"/epoch_"+str(epoch)+".pkl")
                        time.sleep(0.005)
                        pass

                    sio.savemat(path+"/info_fold"+str(rind)+'_nenh'+str(trial)+'.mat', {
                        "train_loss": train_loss_set, "test_loss": test_loss_set, "pred_label": pred_label,'test_label':test_label})
