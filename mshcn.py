import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class mshcn(nn.Module):
    def __init__(self, spec_band, spat_band, num_classes, init_weights=True):
        super(mshcn, self).__init__()

        self.conv0 = nn.Conv2d(spat_band, 64, (1, 1), stride=1, padding=0).float()

        self.conv1 = nn.Conv2d(64, 128, (3, 3), stride=1, padding=1, bias=True).float()
        self.conv2 = nn.Conv2d(spat_band, 128, (1, 1), stride=1, padding=0).float()
        self.conv3 = nn.Conv2d(128, 128, (3, 3), stride=1, padding=0).float()

        self.conv4 = nn.Conv2d(256, 512, (3, 3), stride=1, padding=1, bias=True).float()
        self.conv5 = nn.Conv2d(spat_band, 512, (1, 1), stride=1, padding=0).float()
        self.conv6 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=0).float()

        self.conv7 = nn.Conv2d(1024, 1024, (3, 3), stride=1, padding=1, bias=True).float()
        self.conv8 = nn.Conv2d(spat_band, 1024, (1, 1), stride=1, padding=0).float()
        self.conv9 = nn.Conv2d(1024, 1024, (3, 3), stride=1, padding=0).float()

        self.bn0 = nn.BatchNorm2d(128, track_running_stats=True).float()
        self.bn1 = nn.BatchNorm2d(512, track_running_stats=True).float()
        self.bn2 = nn.BatchNorm2d(1024, track_running_stats=True).float()

        self.bn_fc2 = nn.BatchNorm1d(1024, track_running_stats=True).float()
        self.bn_fc3 = nn.BatchNorm1d(2048, track_running_stats=True).float()

        self.fc0 = nn.Linear(spec_band, 2048, bias=True).float()
        self.fc2 = nn.Linear(2048, 1024, bias=True).float()
        self.fc3 = nn.Linear(4096, 2048, bias=True).float()
        self.fc1 = nn.Linear(1024, num_classes, bias=True).float()

        self.drop2d = nn.Dropout2d(p=0, inplace=True)
        self.drop1 = nn.Dropout(p=0.5, inplace=True)
        #self.drop2 = nn.Dropout(p=0.5, inplace=True)

        self.spat_band = spat_band
        if init_weights:
            self._initialize_weights()

    def forward(self, x_spec, x_spat77_fil):
        # spec
        x1 = self.fc0(x_spec)

        # spat

        x_spat11_fil = x_spat77_fil[:, :, 3:4, 3:4]
        x_spat33_fil = x_spat77_fil[:, :, 2:5, 2:5]
        x_spat55_fil = x_spat77_fil[:, :, 1:6, 1:6]

        ##Filter
        y = self.conv0(x_spat77_fil)  # 7*7*64
        y = F.leaky_relu(self.bn0(self.conv1(y)))  # 7*7*128
        y = torch.cat((self.conv2(x_spat55_fil), self.conv3(y)), 1)  # 5*5*256
        y = self.drop2d(y)

        y = F.leaky_relu(self.bn1(self.conv4(y)))  # 5*5*512
        y = torch.cat((self.conv5(x_spat33_fil), self.conv6(y)), 1)  # 3*3*1024
        y = self.drop2d(y)

        y = F.leaky_relu(self.bn2(self.conv7(y)))  # 3*3*1024
        y = torch.cat((self.conv8(x_spat11_fil), self.conv9(y)), 1)  # 1*1*2048
        y = self.drop2d(y)
        x = y.view(y.size(0), -1)

        # cat,fc
        x = torch.cat((x1, x), 1)
        x = self.drop1(x)
        x = F.relu(self.fc3(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop1(x)
        score = F.log_softmax(self.fc1(x), dim=1)
        return score

    # fork from https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html#vgg11
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                nn.init.constant_(m.bias.data, 0)
class operate():

    def train(self, epoch, loss_trn, net, optimizer, scheduler, trn_loader, criterion):
        net.train()  # train mode
        epochavg_loss = 0
        correct = 0
        total = 0
        for idx, (X_spec, X_spat_fil, y_target) in enumerate(trn_loader):
            X_spec, X_spat_fil = Variable(X_spec.float()).cuda(), Variable(X_spat_fil.float()).cuda()
            ######GPU
            y_target = Variable(y_target.float().long()).cuda()
            y_pred = net.forward(X_spec, X_spat_fil)
            loss = criterion(y_pred, y_target)

            epochavg_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            # print(torch.unique(predicted))
            # print(torch.unique(y_target))
            correct += torch.sum(predicted == y_target)
            total += y_target.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if idx % 20==0:

            del X_spec, X_spat_fil, y_target
            del y_pred
            # del loss
        scheduler.step()
        print('train epoch:{},train loss:{},correct/total:{:.4f}%'.format(epoch,
               epochavg_loss / (idx + 1),100 * correct.item() / total))
        loss_trn.append(epochavg_loss / (idx + 1))
        return loss_trn

    def inference(self,net, data_loader, criterion, FLAG='VAL'):
        net.eval()  # evaluation mode
        inf_loss = 0
        num = 1
        correct = 0
        total = 0
        for idx, (X_spec, X_spat_fil, y_target) in enumerate(data_loader):
            with torch.no_grad():
                X_spec, X_spat_fil = Variable(X_spec.float()).cuda(), Variable(X_spat_fil.float()).cuda()#GPU
                y_target = Variable(y_target.float().long()).cuda()
                y_score = net.forward(X_spec, X_spat_fil)
            loss = criterion(y_score, y_target)
            inf_loss += loss.float()  # save memory

            _, predicted = torch.max(y_score.data, 1)
            correct += torch.sum(predicted == y_target)
            total += y_target.shape[0]

            y_pred_inf = np.argmax(y_score.detach().cpu().numpy(), axis=1) + 1
            if num == 1:
                inf_result = y_pred_inf
            else:
                inf_result = np.hstack((inf_result, y_pred_inf))
            if idx % 20 == 0 and idx > 0:
                print('test loss:{},{}/{}({:.2f}%),correct/total:{:.4f}%'.format(
                    loss.item(), idx * X_spec.shape[0],len(data_loader.dataset),100 * idx * X_spec.shape[0] / len(
                    data_loader.dataset),100 * correct.item() / total))
            num += 1
            del X_spec,X_spat_fil, y_target
            del loss
            del y_score
            del y_pred_inf
        avg_inf_loss = inf_loss / len(data_loader.dataset)
        if FLAG == 'VAL':
            print('Over all validation loss:', inf_loss.cpu().numpy(), 'Average loss:', avg_inf_loss.cpu().numpy(),
                  'correct/total:{:.4f}%'.format(100 * correct.item() / total))
        if FLAG == 'TEST':
            print('Over all testing loss:', inf_loss.cpu().numpy(), 'Average loss:', avg_inf_loss.cpu().numpy(),
                  'correct/total:{:.4f}%'.format(100 * correct.item() / total))
        return inf_result