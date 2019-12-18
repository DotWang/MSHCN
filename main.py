import os
import time
import numpy as np
import matplotlib
#matplotlib.use('module://backend_interagg')
import matplotlib.pyplot as plt
import scipy.io as scio
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler
from func import load,product,preprocess
from mshcn import mshcn, operate
from sklearn.metrics import cohen_kappa_score

USE_GPU=True
if USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    device=torch.device('cpu')

####################################load dataset######################

a=load()
All_data,labeled_data,rows_num,categories,r,c,flag=a.load_data(flag='houston')

np.save('All_data_'+str(flag)+'.npy',All_data)

print('Data has been loaded successfully!')

#################################### dr & norm ######################
# normlization
mi = -1
ma = 1

strategy='ori'#ori/pca/ica

a=preprocess(strategy)
Alldata_DR=a.Dim_reduction(All_data)

a=product(c,flag)
DR_data=a.normlization_2(Alldata_DR,mi,ma)#spec
Origin_data=a.normlization_2(All_data[:,1:-1],mi,ma)#spec

#image_data3D_DR=Alldata_DR.reshape(r,c,-1)

#image_data=All_data[:,1:-1]
#image_data=(image_data_pre-np.min(image_data_pre,axis=0))/(np.max(image_data_pre,axis=0)-np.min(image_data_pre,axis=0))
#image_data3D=image_data.reshape(r,c,-1)
# print(image_data3D.shape)

print('Dimension reduction successfully!')

#################################### data preparing ###################

# image_3d_lr=np.fliplr(image_data3D)
# image_3d_ud=np.flipud(image_data3D)
# image_3d_corner=np.flipud(np.fliplr(image_data3D))
#
# image_3d_temp1=np.hstack((image_3d_corner,image_3d_ud,image_3d_corner))
# image_3d_temp2=np.hstack((image_3d_lr,image_data3D,image_3d_lr))
# image_3d_merge=np.vstack((image_3d_temp1,image_3d_temp2,image_3d_temp1))
#
# image_3d_mat_origin=image_3d_merge[(r-3):2*r+3,(c-3):2*c+3,:]
#
# print(image_3d_mat_origin.shape)
#
# print('image edge enhanced Finished!')
#
# del image_3d_lr,image_3d_ud,image_3d_corner,image_3d_temp1,image_3d_temp2,image_3d_merge
#
# scio.savemat('houston_expand.mat',{'data':image_3d_mat_origin})
#
# print('save finished!')

#load the denoised image

## Gaussian

#image_3d_mat_filter=scio.loadmat('indian_expand_gau_nowaterabsorpt.mat')['data']
image_3d_mat_filter=scio.loadmat('houston_expand_gau.mat')['data']
#image_3d_mat_filter=scio.loadmat('pavia_expand_gau.mat')['data']

print(image_3d_mat_filter.shape)

print('Filtered image import Finished!')

# normlization
mi = -1
ma = 1

a=product(c,flag)
image_3d_mat_filter=a.normlization_2(image_3d_mat_filter,mi,ma)

#################################### Spatial ###########################

Experiment_result=np.zeros([categories+4,12])#OA,AA,kappa,trn_time,test_time，repeat 10 times

Experiment_num=10

#kappa
kappa=0

# half of the patch size
half_s=3

# normlization
mi = -1
ma = 1

for count in range(0,Experiment_num):

    a=product(c,flag)

    rows_num,trn_num,tes_num,pre_num=a.generation_num(labeled_data,rows_num,All_data)

    trn_spat_fil_new, trn_num, _ = a.production_data_trn(rows_num, trn_num, half_s, image_3d_mat_filter)

    tes_spat_fil_new, tes_num = a.production_data_valtespre(tes_num, half_s, image_3d_mat_filter, flag='Tes')
    # ################################### spectral  ############################
    #
    if strategy=='ori':
        # #ORI
        trn_spec = Origin_data[trn_num, :]
        # # val_spec=Origin_data[val_num,:]
        tes_spec = Origin_data[tes_num, :]
        # # pre_spec=Origin_data[pre_num,:]
    else:
        # #DR
        trn_spec = DR_data[trn_num, :]
        # val_spec=DR_data[val_num,:]
        tes_spec = DR_data[tes_num, :]
        # pre_spec=DR_data[pre_num,:]

    #
    # #label
    y_trn=All_data[trn_num,-1]
    # y_val=All_data[val_num,-1]
    y_tes=All_data[tes_num,-1]

    print('trn_spec:',trn_spec.shape)
    # print('val_spec:',val_spec.shape)
    print('tes_spec:',tes_spec.shape)
    # print('pre_spec:',pre_spec.shape)
    # print('Spectral dataset preparation Finished!')
    #
    # ################################### numpy2tensor  ####################################
    #
    trn_XX_spat_fil=torch.from_numpy(trn_spat_fil_new.transpose(0,3,1,2))#(N,C,H,W)
    del trn_spat_fil_new

    tes_XX_spat_fil=torch.from_numpy(tes_spat_fil_new.transpose(0,3,1,2))
    del tes_spat_fil_new

    trn_XX_spec=torch.from_numpy(trn_spec)
    del trn_spec

    tes_XX_spec=torch.from_numpy(tes_spec)
    del tes_spec

    trn_YY=torch.from_numpy(y_trn-1)#label start with 0

    tes_YY=torch.from_numpy(y_tes-1)

    print('Experiments {}, data preparing finished！！'.format(count))

    ####################################### Training ########################################

    torch.cuda.empty_cache()#GPU memory released

    trn_dataset=TensorDataset(trn_XX_spec,trn_XX_spat_fil,trn_YY)
    trn_loader=DataLoader(trn_dataset,batch_size=16,sampler=SubsetRandomSampler(range(trn_XX_spec.shape[0])))

    lr = 1e-3
    epoch = 200

    net = mshcn(trn_XX_spec.shape[1], trn_XX_spat_fil.shape[1], categories-1, init_weights=True)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net = net.cuda()

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90, 120, 150,180], gamma=0.5)

    loss_trn = []
    trn_time1=time.time()

    for i in range(1, epoch):
        a=operate()
        loss_trn = a.train(i, loss_trn, net, optimizer, scheduler, trn_loader, criterion)

    trn_time2=time.time()

    print('Training cost {} sec'.format(trn_time2-trn_time1))

    #print(type(loss_trn))  ######CPU
    plt.figure(1)
    plt.plot(np.array(loss_trn), label='Training')
    plt.legend()
    plt.show()
    #
    # ##save training model
    #torch.save(net,'mshcn.pkl')

    print('Experiments {}，model training finished！！'.format(count))


    #################################### inf, val ################################################
    #
    # val_dataset=TensorDataset(val_XX_spec,val_XX_spat_fil,val_YY)
    # val_loader=DataLoader(val_dataset,batch_size=500)
    #
    # net=torch.load('mshcn.pkl',map_location='cpu')
    # #net=net.module#if use DataParallel
    #
    # net=net.cuda()
    # #net=nn.DataParallel(net,device_ids=[0])
    # #net=net.cpu()
    # a=operate()
    # y_pred_val=a.inference(net,val_loader,criterion,FLAG='VAL')
    #
    # print('Val OA',np.mean(y_pred_val==y_val))

    #################################### inf tes ################################################

    tes_dataset=TensorDataset(tes_XX_spec,tes_XX_spat_fil,tes_YY)
    tes_loader=DataLoader(tes_dataset,batch_size=500)

    # net=torch.load('mshcn.pkl',map_location='cpu')
    # #net=net.module#if use DataParallel
    #
    net=net.cuda()
    #net=nn.DataParallel(net,device_ids=[0])
    #net=net.cpu()
    a=operate()

    tes_time1=time.time()
    y_pred_tes=a.inference(net,tes_loader,criterion,FLAG='TEST')
    tes_time2=time.time()

    print('Testing cost {} sec'.format(tes_time2 - tes_time1))

    #################################### assess #################################################

    ############################### val ###########################

    # print('==================val set=====================')
    # print('Val OA',np.mean(y_val==y_pred_val))
    # print('Val Kappa',cohen_kappa_score(y_val,y_pred_val))
    #
    #
    # num_val=np.zeros([categories-1])
    # num_val_pred=np.zeros([categories-1])
    # for k in y_val:
    #     num_val[k-1]=num_val[k-1]+1
    # for j in range(y_val.shape[0]):
    #     if y_val[j]==y_pred_val[j]:
    #         num_val_pred[y_val[j]-1]=num_val_pred[y_val[j]-1]+1
    #
    # l1=pd.DataFrame(np.arange(categories-1)+1,index=None)
    # l2=pd.DataFrame(num_val_pred/num_val)
    # print(l2.to_string(index=False))

    ############################### tes ###########################

    print('==================Test set=====================')
    print('Experiments {}，Tes OA={}'.format(count,np.mean(y_tes==y_pred_tes)))
    print('Experiments {}，Tes Kappa={}'.format(count,cohen_kappa_score(y_tes,y_pred_tes)))

    if cohen_kappa_score(y_tes, y_pred_tes) >= kappa:
        torch.save(net, 'mshcn_'+strategy+'_' + str(flag) + '.pkl')
        np.save('trn_num_'+strategy+'_' + str(flag) + '.npy', trn_num)
        np.save('pre_num_'+strategy+'_' + str(flag) + '.npy', pre_num)
        np.save('y_trn_'  +strategy+'_' + str(flag) + '.npy', y_trn)
        kappa = cohen_kappa_score(y_tes, y_pred_tes)

    num_tes=np.zeros([categories-1])
    num_tes_pred=np.zeros([categories-1])
    for k in y_tes:
        num_tes[k-1]=num_tes[k-1]+1
    for j in range(y_tes.shape[0]):
        if y_tes[j]==y_pred_tes[j]:
            num_tes_pred[y_tes[j]-1]=num_tes_pred[y_tes[j]-1]+1

    Acc=num_tes_pred/num_tes*100

    Experiment_result[0,count]=np.mean(y_tes==y_pred_tes)*100#OA
    Experiment_result[1,count]=np.mean(Acc)#AA
    Experiment_result[2,count]=cohen_kappa_score(y_tes,y_pred_tes)*100#Kappa
    Experiment_result[3,count]=trn_time2-trn_time1
    Experiment_result[4, count] = tes_time2-tes_time1
    Experiment_result[5:,count]=Acc

    print('Experiments {}, model assessment finished！！!'.format(count))

########## mean std#############

Experiment_result[:,-2]=np.mean(Experiment_result[:,0:-2],axis=1)
Experiment_result[:,-1]=np.std(Experiment_result[:,0:-2],axis=1)

scio.savemat('mshcn_result_'+strategy+'_' +str(flag)+'.mat',{'data':Experiment_result})
np.save('image_3d_mat_origin_'+str(flag)+'.npy',image_3d_mat_filter)

print('Model assessment finished！！')