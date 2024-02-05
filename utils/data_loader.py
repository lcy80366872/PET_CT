import os
import cv2
import torch
import torch.utils.data as data
from .image_augmentation import *
#from .ct_render import ctDataRender, ctImageRender
#from skimage.segmentation import slic

class ImageLidarDataset(data.Dataset):
    def __init__(self, image_list, pet_root, mask_root, lidar_root, pet_suffix="png", mask_suffix="png", lidar_suffix="png", randomize=False, mask_transform=False, adjust_resolution=-1):
        self.image_list = image_list
        
        self.pet_root   = pet_root
        self.mask_root  = mask_root
        self.lidar_root = lidar_root
        
        self.pet_suffix   = pet_suffix
        self.mask_suffix  = mask_suffix
        self.lidar_suffix = lidar_suffix
        
        self.randomize = randomize
        self.mask_transform = mask_transform
        self.adjust_resolution = adjust_resolution
        
    def _read_data(self, image_id):
        img   = cv2.imread(os.path.join(self.pet_root,   "{0}.{1}").format(image_id, self.pet_suffix))   
        mask  = cv2.imread(os.path.join(self.mask_root,  "{0}.{1}").format(image_id, self.mask_suffix),  cv2.IMREAD_GRAYSCALE)
        lidar = cv2.imread(os.path.join(self.lidar_root, "{0}.{1}").format(image_id, self.lidar_suffix), cv2.IMREAD_GRAYSCALE)
        
        assert (img is not None),   os.path.join(self.pet_root,   "{0}.{1}").format(image_id, self.pet_suffix)
        assert (mask is not None),  os.path.join(self.mask_root,  "{0}.{1}").format(image_id, self.mask_suffix)
        assert (lidar is not None), os.path.join(self.lidar_root, "{0}.{1}").format(image_id, self.lidar_suffix)
        
        ## In TLCGIS, the foreground value is 0 and the background value is 1.
        ## The background value is transformed to 0 and the foreground value is transformed to 1 (255)
        if self.mask_transform:
           mask = (1 - mask) * 255

        return img, mask, lidar

    
    def _concat_images(self, image1, image2):
        if image1 is not None and image2 is not None:
            img = np.concatenate([image1, image2], 2)
        elif image1 is None and image2 is not None:
            img = image2
        elif image1 is not None and image2 is None:
            img = image1
        else:
            print("[ERROR] Both images are empty.")
            exit(1)
        return img

    def _data_augmentation(self, pet, mask, lidar):
        if lidar.ndim == 2:
            lidar = np.expand_dims(lidar, axis=2)
            
        if self.randomize:
            pet = randomHueSaturationValue(pet)
            img = self._concat_images(pet, lidar)
            img, mask = randomShiftScaleRotate(img, mask)
            img, mask = randomHorizontalFlip(img, mask)
            img, mask = randomVerticleFlip(img, mask)
            img, mask = randomRotate90(img, mask)
        else:
            img = self._concat_images(pet, lidar)

        if mask.ndim == 2:
           mask = np.expand_dims(mask, axis=2)
        
        # The image's resolution of TLCGIS is 500*500. We change the resolution of input images to 512*512 due to the requirements of network structure.
        # But the resolution of masks is maintained. For a fair comparison, the final predicted maps would be resized to the resolution of masks during testing.
        if self.adjust_resolution > 0:
           img = cv2.resize(img, (self.adjust_resolution, self.adjust_resolution))
            
        try:
            img  = np.array(img,  np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
            mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
        except Exception as e:
            print(e)
            print(img.shape, mask.shape)
        
        mask[mask >= 0.5] = 1
        mask[mask <  0.5] = 0
        return img, mask    

    def __getitem__(self, index):
        image_id = self.image_list[index]
        img, mask, lidar = self._read_data(image_id)
        img, mask = self._data_augmentation(img, mask, lidar)
        img, mask = torch.Tensor(img), torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(self.image_list)



class PET_CT_Dataset(data.Dataset):
    def __init__(self, image_list, pet_root, mask_root, ct_root, pet_suffix="png", mask_suffix="png", ct_suffix="png", randomize=True):
        self.image_list = image_list
        
        self.pet_root = pet_root
        self.mask_root = mask_root
        self.ct_root = ct_root
        
        self.pet_suffix  = pet_suffix
        self.mask_suffix = mask_suffix
        self.ct_suffix  = ct_suffix
        
        self.randomize = randomize

    def _read_data(self, image_id):
        idd=image_id[:4]+'/'+image_id
        pet_path  = os.path.join(self.pet_root,  "{0}_PET.{1}").format(idd,  self.pet_suffix)
        mask_path = os.path.join(self.mask_root, "{0}_CT_mask.{1}").format(idd, self.mask_suffix)
        ct_path  = os.path.join(self.ct_root,  "{0}_CT.{1}").format(idd,  self.ct_suffix)
        
        pet  = cv2.imread(pet_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        ct  = cv2.imread(ct_path,  cv2.IMREAD_GRAYSCALE)
        
        assert (pet is not None),  pet_path
        assert (mask is not None), mask_path
        assert (ct is not None),  ct_path
        
        return pet, mask, ct


    def _concat_images(self, image1, image2):
        if image1 is not None and image2 is not None:
            img = np.concatenate([image1, image2], 2)
        elif image1 is None and image2 is not None:
            img = image2
        elif image1 is not None and image2 is None:
            img = image1
        else:
            print("[ERROR] Both images are empty.")
            exit(1)
        return img

    def _data_augmentation(self, pet, mask, ct):
        if pet.ndim == 2:
            pet = np.expand_dims(pet, axis=2)
        if ct.ndim == 2:
            ct = np.expand_dims(ct, axis=2)
            
        if self.randomize:
            img = self._concat_images(pet, ct)
            img, mask = randomShiftScaleRotate(img, mask)
            img, mask = randomHorizontalFlip(img, mask)
            img, mask = randomcrop(img,mask)
        else:
            img = self._concat_images(pet, ct)
    

        if mask.ndim == 2:
           mask = np.expand_dims(mask, axis=2)

        try:
            img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
            mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
        except Exception as e:
            print(e)
            print(img.shape, mask.shape)

        mask[mask >= 0.5] = 1
        mask[mask <  0.5] = 0
        return img, mask


    def __getitem__(self, index):
        image_id = self.image_list[index]
        pet, mask, ct= self._read_data(image_id)
        pet, mask = self._data_augmentation(pet, mask, ct)
        pet, mask = torch.Tensor(pet), torch.Tensor(mask)
        return pet, mask

    def __len__(self):
        return len(self.image_list)

