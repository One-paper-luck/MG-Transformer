import os
import numpy as np
import torch
import torch.nn.functional as F
import random
random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

def readInfo(filePath):
    name = os.listdir(filePath)
    return name

def data_normal(data):
    d_min=data.min()
    if d_min<0:
        data+=torch.abs(d_min)
        d_min=data.min()

    d_max=data.max()
    dst=d_max-d_min
    nor_data=(data-d_min).true_divide(dst)
    return nor_data


features_path='/media/dmd/ours/mlw/rs/multi_scale_T/sydney_res152_con5_224'
out_path='/media/dmd/ours/mlw/rs/multi_scale_T/sydney_group_mask_8_6'

num=6



fileList = readInfo(features_path)

for i, img in enumerate(fileList):
        res_feature=np.load(features_path + '/%s' % img)
        r_feature = res_feature.reshape(res_feature.shape[0], res_feature.shape[1] * res_feature.shape[2])
        r_feature=torch.Tensor(r_feature)

        res_feature=torch.Tensor(res_feature)
        avg_feature = F.adaptive_avg_pool2d(res_feature, [1, 1]).squeeze()
        avg_feature=avg_feature.unsqueeze(0)
        p=torch.matmul(avg_feature, r_feature)

        pp=data_normal(p)

        # divide group  8
        group_mask = torch.ones((8, 49, 49)).cuda() # res

        index_1 = torch.nonzero(pp <= 1 / 8, as_tuple=False)[:, 1]
        index_2 = torch.nonzero((pp <= 2 / 8) & (pp > 1 / 8), as_tuple=False)[:, 1]
        index_3 = torch.nonzero((pp <= 3 / 8) & (pp > 2 / 8), as_tuple=False)[:, 1]
        index_4 = torch.nonzero((pp <= 4 / 8) & (pp > 3 / 8), as_tuple=False)[:, 1]
        index_5 = torch.nonzero((pp <= 5 / 8) & (pp > 4 / 8), as_tuple=False)[:, 1]
        index_6 = torch.nonzero((pp <= 6 / 8) & (pp > 5 / 8), as_tuple=False)[:, 1]
        index_7 = torch.nonzero((pp <= 7 / 8) & (pp > 3 / 8), as_tuple=False)[:, 1]
        index_8 = torch.nonzero((pp <= 8 / 8) & (pp > 7 / 8), as_tuple=False)[:, 1]



        for k in range(8):
            list = random.sample(range(1, 9), num)
            for j in range(2):
                if list[j] == 1:
                    length = index_1.__len__()
                    for i in range(length):
                        group_mask[k][:, index_1[i]] = 0
                elif list[j] == 2:
                    length = index_2.__len__()
                    for i in range(length):
                        group_mask[k][:, index_2[i]] = 0
                elif list[j] == 3:
                    length = index_3.__len__()
                    for i in range(length):
                        group_mask[k][:, index_3[i]] = 0
                elif list[j] == 4:
                    length = index_4.__len__()
                    for i in range(length):
                        group_mask[k][:, index_4[i]] = 0
                elif list[j] == 5:
                    length = index_5.__len__()
                    for i in range(length):
                        group_mask[k][:, index_5[i]] = 0
                elif list[j] == 6:
                    length = index_6.__len__()
                    for i in range(length):
                        group_mask[k][:, index_6[i]] = 0
                elif list[j] == 7:
                    length = index_7.__len__()
                    for i in range(length):
                        group_mask[k][:, index_7[i]] = 0
                elif list[j] == 8:
                    length = index_8.__len__()
                    for i in range(length):
                        group_mask[k][:, index_8[i]] = 0

        np.save(os.path.join(out_path, img), group_mask.data.cpu().float().numpy()) # size: 8，49，49

