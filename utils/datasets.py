import os
from sklearn.model_selection import train_test_split
from .data_loader import  PET_CT_Dataset, ImageLidarDataset


def get_all_file_and_path(rootpath, allfilelist, dirfilelist):
    dir_or_file = os.listdir(rootpath)

    for dirfile in dir_or_file:

        # print(dirfile)
        dirpathnew = os.path.join(rootpath, dirfile)

        if os.path.isdir(dirpathnew):
            dirfilelist.append(dirpathnew)

            get_all_file_and_path(dirpathnew, allfilelist, dirfilelist)
        else:
            allfilelist.append(dirpathnew)

    return allfilelist, dirfilelist


def prepare_PETCT_dataset(args):
    print("")
    print("Dataset: ", args.dataset)

        
    print("")
    print("pet_dir: ", args.pet_dir)
    print("ct_dir: ", args.ct_dir)
    print("mask_dir: ", args.mask_dir)
    image_list=[]
    for root, folders, files in os.walk(args.pet_dir):
        for x in files:
            if x.find('PET.png') != -1:
                image_list.append(x[:-8])
    train_list, test_list = train_test_split(image_list, test_size=args.val_size, random_state=args.random_seed)
    train_dataset =  PET_CT_Dataset(train_list, args.pet_dir,      args.mask_dir,      args.ct_dir,      randomize=True)
    val_dataset = PET_CT_Dataset(test_list, args.pet_dir, args.mask_dir, args.ct_dir, randomize=False)
    test_dataset  =  PET_CT_Dataset(test_list,  args.pet_dir, args.mask_dir,  args.ct_dir, randomize=False)

    return train_dataset, val_dataset, test_dataset
    
    
def prepare_TLCGIS_dataset(args):
    print("")
    print("Dataset: ", args.dataset)
    mask_transform = True if args.dataset == 'TLCGIS' else False
    adjust_resolution =512 if args.dataset == 'TLCGIS' else -1
    
    print("")
    print("sat_dir: ", args.sat_dir)
    print("gps_dir: ", args.lidar_dir)    
    print("mask_dir: ", args.mask_dir)
    print("partition_txt: ", args.split_train_val_test)
    print("mask_transform: ", mask_transform)
    print("adjust_resolution: ", adjust_resolution)
    print("")
        
    train_list = val_list = test_list = []
    with open(os.path.join(args.split_train_val_test,'train.txt'),'r') as f:
        train_list = [x[:-1] for x in f]
    with open(os.path.join(args.split_train_val_test,'valid.txt'),'r') as f:
        val_list = [x[:-1] for x in f]
    with open(os.path.join(args.split_train_val_test,'test.txt'),'r') as f:
        test_list = [x[:-1] for x in f]

    train_dataset = ImageLidarDataset(train_list, args.sat_dir, args.mask_dir, args.lidar_dir, randomize=False,  mask_transform=mask_transform, adjust_resolution=adjust_resolution)
    val_dataset   = ImageLidarDataset(val_list,   args.sat_dir, args.mask_dir, args.lidar_dir, randomize=False, mask_transform=mask_transform, adjust_resolution=adjust_resolution)
    test_dataset  = ImageLidarDataset(test_list,  args.sat_dir, args.mask_dir, args.lidar_dir, randomize=False, mask_transform=mask_transform, adjust_resolution=adjust_resolution)

    return train_dataset, val_dataset, test_dataset


