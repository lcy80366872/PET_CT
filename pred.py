import cv2
import os
import numpy as np
import torch

import torch.nn as nn
from tqdm import tqdm
#from scipy.ndimage.morphology import distance_transform_edt
from networks.exchange_dlink34net import DinkNet34_CMMPNet


pet_root = 'E:/ML_data/PCLC1-108/PET/'
ct_root= 'E:/ML_data/PCLC1-108/CT/'
test_mask_dir='E:/ML_data/PCLC1-108/CT/'
# 输出目录
outPath ='E:/ML_data/PCLC1-108/pred/'
image_list=[]
for root, folders, files in os.walk(pet_root):
    for x in files:
        if x.find('PET.png') != -1:
            image_list.append(x[:-8])


def get_model(model_name):
    if model_name == 'CMMPNet':
        model = DinkNet34_CMMPNet()
    else:
        print("[ERROR] can not find model ", model_name)
        assert(False)
    return model
net = get_model('CMMPNet')  #exchange_pol1_epoch28_val0.7234_test0.6384
state_dict = torch.load('save_model/epoch60_val0.8056_test0.8056.pth', map_location=torch.device('cpu'))
net.load_state_dict(state_dict)
#
# n=0
# for name, param in net.named_parameters():
#     if param.requires_grad and name.endswith('weight') and 'bn2' in name:
#         a=param.detach()
#         n=n+1
#         print('第n层：',n)
#         # print(a)
#         # if n==25 or n==26:
#         print(list(np.array(a)))


# new_state = {}
# for key, value in state_dict.items():
#     new_state[key.replace('module.', '')] = value
# net.load_state_dict(new_state)
def processImage(path1, path2,net,destsource, name):
    net.eval()
    img = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    print(name)
    img1 = np.expand_dims(img1, axis=2)
    img = np.expand_dims(img, axis=2)
    img = np.concatenate([img, img1], axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    img = img[np.newaxis, :, :, :]
    img = torch.tensor(img).cuda()
    net.cuda()
    with torch.no_grad():
        pred = net.forward(img)
    pred = pred.cpu().numpy()

    pred = np.squeeze(pred, axis=0).transpose(1, 2, 0)*255
    cv2.imwrite(destsource + name+'.png', pred)

if __name__ == '__main__':
    print('start')
    image_list=['0355_007']
    for i in image_list:
        idd = i[:4] + '/' + i

        pet_path = os.path.join(pet_root, "{0}_PET.{1}").format(idd, "png")
        ct_path = os.path.join(ct_root, "{0}_CT.{1}").format(idd, "png")
        print(pet_path)
        processImage(pet_path,ct_path, net,outPath, i)
