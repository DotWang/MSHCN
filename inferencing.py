import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from mshcn import mshcn, operate
from func import load,product,preprocess

USE_GPU=True
if USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    device=torch.device('cpu')
print('using device:','0,1')

a=load()

All_data, _, _, _, r, c, flag=a.load_data(flag='indian')

###################### load data/model, parameters setting and global variable ####################
# spectral normalization（Due to load image_3d_mat_origin，not need to norm spatial part in inference）
mi = -1
ma = 1

# half of the patch size
half_s=3

strategy='ori'#ica/pca/ori

a=preprocess(strategy)
Alldata_DR=a.Dim_reduction(All_data)

a=product(c,flag)
DR_data=a.normlization_2(Alldata_DR,mi,ma)#spec
Origin_data=a.normlization_2(All_data[:,1:-1],mi,ma)#spec

# load the saving files

trn_num=np.load('trn_num_'+strategy+'_' +str(flag)+'.npy')
pre_num=np.load('pre_num_'+strategy+'_' +str(flag)+'.npy')
y_trn=np.load('y_trn_'+strategy+'_' +str(flag)+'.npy')
image_3d_mat_origin=np.load('image_3d_mat_origin_'+str(flag)+'.npy')

### config the net

net=torch.load('mshcn_'+strategy+'_' + str(flag) + '.pkl',map_location='cpu')
criterion = torch.nn.NLLLoss()

#net=net.module#if use DataParallel

net=net.cuda()

###### label setting
y_disp=np.zeros([All_data.shape[0]])
y_disp[trn_num]=y_trn

y_disp_all=y_disp.copy()

###### bunch setting #######
BATCH=50000

start=0

end=np.min([start+BATCH,pre_num.shape[0]])

part_num=int(pre_num.shape[0]/BATCH)+1

print('Need to be devided into {} parts'.format(part_num))

for i in range(0,part_num):

    pre_num_part=pre_num[start:end]

    ### label

    y_pre=All_data[pre_num_part,-1]#include background

    pre_YY = torch.from_numpy(np.ones([y_pre.shape[0]]))

    ### spec
    if strategy=='ori':
        pre_spec = Origin_data[pre_num_part, :]
    else:
        pre_spec = DR_data[pre_num_part, :]

    pre_XX_spec = torch.from_numpy(pre_spec)

    ### spat
    a = product(c, flag)

    pre_spat, pre_num_part = a.production_data_valtespre(pre_num_part, half_s, image_3d_mat_origin, flag='Pre')

    pre_XX_spat_fil=torch.from_numpy(pre_spat.transpose(0, 3, 1, 2))

    #################################### inf，pre ################################################

    pre_dataset=TensorDataset(pre_XX_spec,pre_XX_spat_fil,pre_YY)
    pre_loader=DataLoader(pre_dataset,batch_size=500)

    #net=nn.DataParallel(net,device_ids=[0])
    #net=net.cpu()
    a=operate()
    y_pred_pre=a.inference(net,pre_loader,criterion,FLAG='PRED')

    print('Data of part {} prediction finished！！！'.format(i))

    y_disp_all[pre_num_part]=y_pred_pre

    start=end

    end=np.min([start+BATCH,pre_num.shape[0]])

#################################### show & save #################################################

plt.subplots(figsize=[10,10])

a1=plt.imshow(y_disp_all.reshape(r,c),cmap='jet')
plt.xticks([])
plt.yticks([])
plt.savefig('mscn_'+strategy+'_all_'+str(flag)+'.png',dpi=600,bbox_inches='tight')
#plt.show()

print('Classification map saving finshed！！')