import numpy as np
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

class load():
    # load dataset(indian_pines & pavia_univ & houston.)
    def load_data(self,flag='indian'):
        if flag == 'indian':
            Ind_pines_dict = scio.loadmat('/data/di.wang/ordinary/23DCNN/Indian_pines.mat')
            Ind_pines_gt_dict = scio.loadmat('/data/di.wang/ordinary/23DCNN/Indian_pines_gt.mat')

            print(Ind_pines_dict['indian_pines'].shape)
            print(Ind_pines_gt_dict['indian_pines_gt'].shape)

            # remove the water absorption bands

            no_absorption = list(set(np.arange(0, 103)) | set(np.arange(108, 149)) | set(np.arange(163, 219)))

            original = Ind_pines_dict['indian_pines'][:, :, no_absorption].reshape(145 * 145, 200)

            print(original.shape)
            print('Remove wate absorption bands successfully!')

            gt = Ind_pines_gt_dict['indian_pines_gt'].reshape(145 * 145, 1)

            r = Ind_pines_dict['indian_pines'].shape[0]
            c = Ind_pines_dict['indian_pines'].shape[1]
            categories = 17
        if flag == 'pavia':  # too big to test
            pav_univ_dict = scio.loadmat('/data/di.wang/ordinary/23DCNN/PaviaU.mat')
            pav_univ_gt_dict = scio.loadmat('/data/di.wang/ordinary/23DCNN/PaviaU_gt.mat')

            print(pav_univ_dict['paviaU'].shape)
            print(pav_univ_gt_dict['paviaU_gt'].shape)

            original = pav_univ_dict['paviaU'].reshape(610 * 340, 103)
            gt = pav_univ_gt_dict['paviaU_gt'].reshape(610 * 340, 1)

            r = pav_univ_dict['paviaU'].shape[0]
            c = pav_univ_dict['paviaU'].shape[1]
            categories = 10
        if flag == 'houston':
            houst_dict = scio.loadmat('/data/di.wang/ordinary/23DCNN/Houston.mat')
            houst_gt_dict = scio.loadmat('/data/di.wang/ordinary/23DCNN/Houston_GT.mat')

            print(houst_dict['Houston'].shape)
            print(houst_gt_dict['Houston_GT'].shape)

            original = houst_dict['Houston'].reshape(349 * 1905, 144)
            gt = houst_gt_dict['Houston_GT'].reshape(349 * 1905, 1)

            r = houst_dict['Houston'].shape[0]
            c = houst_dict['Houston'].shape[1]
            categories = 16

        rows = np.arange(gt.shape[0])  # start with 0
        # row number(ID)，data，gt
        All_data = np.c_[rows, original, gt]

        # all labeled data
        labeled_data = All_data[All_data[:, -1] != 0, :]
        rows_num = labeled_data[:, 0]  # id of all labeled data

        return All_data, labeled_data, rows_num, categories, r, c, flag

class product():
    def __init__(self,c,flag):
        self.c=c
        self.flag=flag
    # product the training and testing pixel ID
    def generation_num(self,labeled_data, rows_num, All_data):

        train_num = []

        for i in np.unique(labeled_data[:, -1]):
            temp = labeled_data[labeled_data[:, -1] == i, :]
            temp_num = temp[:, 0]
            print(i, temp_num.shape[0])
            np.random.shuffle(temp_num)
            if self.flag == 'indian':
                if i == 1:
                    train_num.append(temp_num[0:33])
                elif i == 7:
                    train_num.append(temp_num[0:20])
                elif i == 9:
                    train_num.append(temp_num[0:14])
                elif i == 16:
                    train_num.append(temp_num[0:75])
                else:
                    train_num.append(temp_num[0:100])
            if self.flag == 'pavia':
                train_num.append(temp_num[0:100])
            if self.flag == 'houston':
                train_num.append(temp_num[0:25])

        trn_num = [x for j in train_num for x in j]
        tes_num = list(set(rows_num) - set(trn_num))
        pre_num = list(set(range(0, All_data.shape[0])) - set(trn_num))
        print('number of training sample', len(trn_num))
        return rows_num, trn_num, tes_num, pre_num


    def production_data(self, rows_num, trn_num, tes_num, pre_num,image_3d_mat):
        trn_num = np.array(trn_num)
        ##Training set(spatial)
        idx_2d_trn = np.zeros([trn_num.shape[0], 2]).astype(int)
        idx_2d_trn[:, 0] = np.floor(trn_num / self.c)
        idx_2d_trn[:, 1] = trn_num + 1 - self.c * idx_2d_trn[:, 0] - 1
        # neibour area(7*7)
        trn_spat = np.zeros([trn_num.shape[0], 7, 7, image_3d_mat.shape[2]])
        neibour_num = []
        for i in range(idx_2d_trn.shape[0]):
            # image expandiation
            row = idx_2d_trn[i, 0] + 3
            col = idx_2d_trn[i, 1] + 3
            trn_spat[i, :, :, :] = image_3d_mat[(row - 3):row + 4, (col - 3):col + 4, :]
            # map the pixel in expandiation image 2 original
            neibour_num = neibour_num + [(row + j - 3) * image_3d_mat.shape[1] + col + k - 3 for j in range(-3, 4) for k
                                         in range(-3, 4)]
        val_num = list(set(rows_num) - set(neibour_num))  # prevent data snooping

        ##Validation set(spatial): accuracy assessment
        val_num = np.array(val_num)
        idx_2d_val = np.zeros([val_num.shape[0], 2]).astype(int)
        idx_2d_val[:, 0] = np.floor(val_num / self.c)
        idx_2d_val[:, 1] = val_num + 1 - self.c * idx_2d_val[:, 0] - 1
        val_spat = np.zeros([val_num.shape[0], 7, 7, image_3d_mat.shape[2]])
        for i in range(idx_2d_val.shape[0]):
            # image expandiation
            row = idx_2d_val[i, 0] + 3
            col = idx_2d_val[i, 1] + 3
            val_spat[i, :, :, :] = image_3d_mat[(row - 3):row + 4, (col - 3):col + 4, :]

        ##Testing set(spatial)
        tes_num = np.array(tes_num)
        idx_2d_tes = np.zeros([tes_num.shape[0], 2]).astype(int)
        idx_2d_tes[:, 0] = np.floor(tes_num / self.c)
        idx_2d_tes[:, 1] = tes_num + 1 - self.c * idx_2d_tes[:, 0] - 1
        tes_spat = np.zeros([tes_num.shape[0], 7, 7, image_3d_mat.shape[2]])
        for i in range(idx_2d_tes.shape[0]):
            # image expandiation
            row = idx_2d_tes[i, 0] + 3
            col = idx_2d_tes[i, 1] + 3
            tes_spat[i, :, :, :] = image_3d_mat[(row - 3):row + 4, (col - 3):col + 4, :]

        ##Predicting set(spatial)
        pre_num = np.array(pre_num)
        idx_2d_pre = np.zeros([pre_num.shape[0], 2]).astype(int)
        idx_2d_pre[:, 0] = np.floor(pre_num / self.c)
        idx_2d_pre[:, 1] = pre_num + 1 - self.c * idx_2d_pre[:, 0] - 1
        pre_spat = np.zeros([pre_num.shape[0], 7, 7, image_3d_mat.shape[2]])
        for i in range(idx_2d_pre.shape[0]):
            # image expandiation
            row = idx_2d_pre[i, 0] + 3
            col = idx_2d_pre[i, 1] + 3
            pre_spat[i, :, :, :] = image_3d_mat[(row - 3):row + 4, (col - 3):col + 4, :]

        print('trn_spat:', trn_spat.shape)
        print('val_spat:', val_spat.shape)
        print('tes_spat:', tes_spat.shape)
        print('pre_spat:', pre_spat.shape)
        print('Spatial dataset preparation Finished!')

        return trn_spat, val_spat, tes_spat, pre_spat, trn_num, val_num, tes_num, pre_num

    def production_data_trn(self, rows_num, trn_num, half_s, image_3d_mat):

        trn_num = np.array(trn_num)
        ##Training set(spatial)
        idx_2d_trn = np.zeros([trn_num.shape[0], 2]).astype(int)
        idx_2d_trn[:, 0] = np.floor(trn_num / self.c)
        idx_2d_trn[:, 1] = trn_num + 1 - self.c * idx_2d_trn[:, 0] - 1
        # neibour area(2*half_s+1)
        patch_size=2*half_s+1
        trn_spat = np.zeros([trn_num.shape[0], patch_size, patch_size, image_3d_mat.shape[2]])
        neibour_num = []
        for i in range(idx_2d_trn.shape[0]):
            # image expandiation
            row = idx_2d_trn[i, 0] + half_s
            col = idx_2d_trn[i, 1] + half_s
            trn_spat[i, :, :, :] = image_3d_mat[(row - half_s):row + half_s + 1,
                                   (col - half_s):col + half_s + 1, :]
            # map the pixel in expandiation image 2 original
            neibour_num = neibour_num + [(row + j - half_s) * self.c + col + k - half_s for j in range(-half_s, half_s+1) for k
                                         in range(-half_s, half_s+1)]
        val_num = list(set(rows_num) - set(neibour_num))  # prevent data snooping

        print('trn_spat:', trn_spat.shape)
        print('Training Spatial dataset preparation Finished!')
        return trn_spat, trn_num, val_num

    def production_data_valtespre(self, tes_num, half_s, image_3d_mat, flag='Tes'):

        ##Testing set(spatial)
        tes_num = np.array(tes_num)
        idx_2d_tes = np.zeros([tes_num.shape[0], 2]).astype(int)
        idx_2d_tes[:, 0] = np.floor(tes_num / self.c)
        idx_2d_tes[:, 1] = tes_num + 1 - self.c * idx_2d_tes[:, 0] - 1
        # neibour area(2*half_s+1)
        patch_size = 2 * half_s + 1
        tes_spat = np.zeros([tes_num.shape[0], patch_size, patch_size, image_3d_mat.shape[2]])
        for i in range(idx_2d_tes.shape[0]):
            # image expandiation
            row = idx_2d_tes[i, 0] + half_s
            col = idx_2d_tes[i, 1] + half_s
            tes_spat[i, :, :, :] = image_3d_mat[(row - half_s):row + half_s + 1,
                                   (col - half_s):col + half_s + 1, :]

        print('tes_spat:', tes_spat.shape)
        print('{} Spatial dataset preparation Finished!'.format(flag))
        return tes_spat,tes_num

    def normlization_1(self,trn_spat, val_spat, tes_spat, pre_spat):

        scaler = MinMaxScaler(feature_range=(-1, 1))

        all_spat_3d = np.concatenate((trn_spat, val_spat, tes_spat, pre_spat), axis=0)
        all_spat_data = all_spat_3d.reshape(-1, trn_spat.shape[-1])
        all_spat_3d_new = scaler.fit_transform(all_spat_data).reshape(all_spat_3d.shape)

        trn_spat_new = all_spat_3d_new[:trn_spat.shape[0], :, :, :]
        val_spat_new = all_spat_3d_new[trn_spat.shape[0]:trn_spat.shape[0] + val_spat.shape[0], :, :, :]
        tes_spat_new = all_spat_3d_new[-tes_spat.shape[0] - pre_spat.shape[0]:-pre_spat.shape[0], :, :, :]
        pre_spat_new = all_spat_3d_new[-pre_spat.shape[0]:, :, :, :]

        print('trn_spat:', trn_spat_new.shape)
        print('val_spat:', val_spat_new.shape)
        print('tes_spat:', tes_spat_new.shape)
        print('pre_spat:', pre_spat_new.shape)
        print('Spatial dataset normalization Finished!')

        return trn_spat_new, val_spat_new, tes_spat_new, pre_spat_new

    def normlization_2(self, data_spat, mi, ma, flag='trn'):

        scaler = MinMaxScaler(feature_range=(mi, ma))

        spat_data = data_spat.reshape(-1, data_spat.shape[-1])
        data_spat_new = scaler.fit_transform(spat_data).reshape(data_spat.shape)

        print('{}_spat:{}'.format(flag,data_spat_new.shape))
        print('{} Spatial dataset normalization Finished!'.format(flag))
        return data_spat_new


class preprocess():
    def __init__(self,t):
        self.transform=t
    def Dim_reduction(self, All_data):

        Alldata_DR=All_data

        if self.transform =='ica':
            ica_data_pre = All_data[:, 1:-1]
            print(ica_data_pre.shape)
            transformer = FastICA(n_components=50, whiten=True, random_state=None)
            fastica_data = transformer.fit_transform(ica_data_pre)
            print(fastica_data.shape)

            Alldata_DR = fastica_data

            print('ICA Finished!')

        if self.transform =='pca':
            pca_data_pre = All_data[:, 1:-1]
            print(pca_data_pre.shape)
            pca_transformer = PCA(n_components=50)
            pca_data = pca_transformer.fit_transform(All_data[:, 1:-1])
            print(pca_data.shape)

            Alldata_DR = pca_data

            print('PCA Finished!')

        return Alldata_DR