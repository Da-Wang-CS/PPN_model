# Carlos X. Soto, csoto@bnl.gov, 2022

import torch
#from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import datetime
from tqdm.auto import tqdm
import itertools

from model.Datasets import BarDataset, ImageOnlyDataset

#checkpoint_dir = 'checkpoints'

#valid_datasets = ('augmented_bars', 'generated_bars',
#                  'generated_pies', 'manually_annotated',
#                  'zhou_2021')

"""

# Custom padding function (resolve batch size error)
def padding(data):
    #print(len(data[0][2]))

    
    for idx in range(len(data[1][2])):
        datatype = type(data[1][2][idx])
        print(datatype)
    

    data[0] = list(data[0])
    print(type(data[0]))
    
    
    largelen = 0

    for b in range(len(data)):
        if len(data[b][2][4]) > largelen:
            largelen = len(data[b][2][4])

    for b in range(len(data)):
        if len(data[b][2][4]) != largelen:

            for a in range(largelen - len(data[b][2][4])):
                data[b][2][4].extend([torch.tensor(0.), torch.tensor(0.)])
                print(data[b][2][4])

            print(len(data[b][2][4]))

            
            while len(data[b][2][4]) != largelen:
                data[b][2][4].append([torch.tensor(0.), torch.tensor(0.)])
            
    
    largelen = 0

    for c in range(len(data)):
        if len(data[c][2][5]) > largelen:
            largelen = len(data[c][2][5])

    for c in range(len(data)):
        if len(data[c][2][5]) != largelen:
            
            for a in range(largelen - len(data[c][2][5])):
                data[c][2][5].extend([torch.tensor(0.), torch.tensor(0.)])           

    
            while len(data[c][2][5]) != largelen:
                data[c][2][5].append([torch.tensor(0.), torch.tensor(0.)])
    

    for a in range(len(data)):    
        tmpPad = list(data[a][2][:-2])
        print(tmpPad[2].shape)
        tmpPad = pad_sequence(tmpPad, batch_first = True)
        data[a][2] = (tmpPad[0], tmpPad[1], tmpPad[2], tmpPad[3], data[a][2][4], data[a][2][5])

    for a in range(len(data)):
        tmpconv = data
        tmpconv[a] = list(tmpconv[a])
        print(type(tmpconv[a]))
        data[a][2] = list(data[a][2])
        
        data[a][2][4] = itertools.zip_longest(data[a][2][4], fillvalue = [torch.tensor(0.), torch.tensor(0.)])
        data[a][2][5] = itertools.zip_longest(data[a][2][5], fillvalue = [torch.tensor(0.), torch.tensor(0.)])
        
        # data[a][2] = tuple(data[a][2]) 

    for b in range(len(data)):
        print(len(data))

    return data

"""

def get_dataloaders(dataset_id, bs) -> torch.utils.data.dataloader.DataLoader:
    #assert dataset_id in valid_datasets, 'invalid dataset ID'
    
    im_path = f'datasets/{dataset_id}/imgs'
    annot_path = f'datasets/{dataset_id}/annots'
    train_list = f'datasets/{dataset_id}/train.txt'
    val_list = f'datasets/{dataset_id}/val.txt'

    train_set = BarDataset(train_list, im_path, annot_path)
    val_set = BarDataset(val_list, im_path, annot_path)

    #train_set = padding(train_set)
    #val_set = padding(val_set)
    
    #train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = bs, num_workers=8, shuffle = True, collate_fn = pad_collate)
    #val_dataloader = torch.utils.data.DataLoader(val_set, batch_size = bs, num_workers=8, shuffle = False, collate_fn = pad_collate)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = bs, num_workers=8, shuffle = True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size = bs, num_workers=8, shuffle = False)

    return train_dataloader, val_dataloader

'''
# Dataset and dataloader
train_list = 'datasets/generated_bars/train.txt'
test_list = 'datasets/generated_bars/test.txt'
im_path = 'datasets/generated_bars/imgs'
annot_path = 'datasets/generated_bars/annots'

# new, isolated testset (not to be used for HPO)
test2_im_path = 'generated_plots/testset_plots'
test2_annot_path = 'generated_plots/testset_annots'
test2_list = 'generated_plots/test2.txt'

# un-annotated bio images
bio_impath = 'yolo/data/biocharts/images'
bio_imlist = 'yolo/data/biocharts/imagelist_nameonly.txt'

# annotated bio images (N=95)
bioannot_impath = 'annotated_plots/v2_246charts/imgs246'
bioannot_annotpath = 'annotated_plots/v2_246charts/m246_annots'
bioannot_train = 'annotated_plots/v2_246charts/m246_train.txt'
bioannot_test = 'annotated_plots/v2_246charts/m246_test.txt'

# real-augmented synthetic plots
real_aug_impath = 'real-augmented_plots/plots'
real_aug_annotpath = 'real-augmented_plots/annots'
real_aug_train = 'real-augmented_plots/train.txt'
real_aug_val = 'real-augmented_plots/val.txt'

# Zhao bar chart dataset
z_impath = 'zhao_dataset/imgs'
z_annotpath = 'zhao_dataset/annots'
z_trainfull = 'zhao_dataset/train.txt'
z_train = 'zhao_dataset/train4k.txt'
z_test = 'zhao_dataset/test.txt'


# datasets
train_set = BarDataset(dataset_list, im_path, annot_path)
test_set = BarDataset(test_list, im_path, annot_path)
bio_set = ImageOnlyDataset(bio_imlist, bio_impath)
test2_set = BarDataset(test2_list, test2_im_path, test2_annot_path)

train4_set = BarDataset(bioannot_train, bioannot_impath, bioannot_annotpath)
test4_set = BarDataset(bioannot_test, bioannot_impath, bioannot_annotpath)

train_aug_set = BarDataset(real_aug_train, real_aug_impath, real_aug_annotpath)
val_aug_set = BarDataset(real_aug_val, real_aug_impath, real_aug_annotpath)

train_z_set = BarDataset(z_train, z_impath, z_annotpath)
test_z_set = BarDataset(z_test, z_impath, z_annotpath)


# batch size
bs = 1

# dataloaders
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = bs, num_workers=8, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = bs, num_workers=8, shuffle = True)
bioimg_dataloader = torch.utils.data.DataLoader(bio_set, batch_size = bs, num_workers=8, shuffle = True)
test2_dataloader = torch.utils.data.DataLoader(test2_set, batch_size = bs, num_workers=8, shuffle = False)

train4_dataloader = torch.utils.data.DataLoader(train4_set, batch_size = bs, num_workers=8, shuffle = False)
test4_dataloader = torch.utils.data.DataLoader(test4_set, batch_size = bs, num_workers=8, shuffle = False)

train_aug_dataloader = torch.utils.data.DataLoader(train_aug_set, batch_size = bs, num_workers=8, shuffle = False)
val_aug_dataloader = torch.utils.data.DataLoader(val_aug_set, batch_size = bs, num_workers=8, shuffle = False)

train_z_dataloader = torch.utils.data.DataLoader(train_z_set, batch_size = bs, num_workers=8, shuffle = True)
test_z_dataloader = torch.utils.data.DataLoader(test_z_set, batch_size = bs, num_workers=8, shuffle = False)
'''


# LOG LOSSES AND BATCH TRAINING TIMES

def log_losses(start_epoch, num_epochs,
               origin_losses, pclass_losses, pntreg_losses, losses, train_times,
               vorigin_losses, vpclass_losses, vpntreg_losses, vlosses, vtimes,
               barPs, barRs, barF1s, tickPs, tickRs, tickF1s, euPs, euRs, euF1s, edPs, edRs, edF1s, meanF1s):
    timestring = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfn = f'logs/{timestring}_train_epochs_{start_epoch}-{num_epochs}.log'
    with open(logfn,'w') as logfile:
        logfile.write('Training origin losses:\n')
        logfile.write(', '.join([str(item) for item in origin_losses]))
    #    logfile.write('\nTraining orient losses:\n')
    #    logfile.write(', '.join([str(item) for item in orient_losses]))
        logfile.write('\nTraining point classification losses:\n')
        logfile.write(', '.join([str(item) for item in pclass_losses]))
        logfile.write('\nTraining point regression losses:\n')
        logfile.write(', '.join([str(item) for item in pntreg_losses]))
        logfile.write('\nTraining total losses:\n')
        logfile.write(', '.join([str(item) for item in losses]))
        logfile.write('\nTraining times:\n')
        logfile.write(', '.join([str(item) for item in train_times]))
        
        logfile.write('\nTest origin losses:\n')
        logfile.write(', '.join([str(item) for item in vorigin_losses]))
        logfile.write('\nTest orient losses:\n')
    #    logfile.write(', '.join([str(item) for item in vorient_losses]))
    #    logfile.write('\nTest point classification losses:\n')
        logfile.write(', '.join([str(item) for item in vpclass_losses]))
        logfile.write('\nTest point regression losses:\n')
        logfile.write(', '.join([str(item) for item in vpntreg_losses]))
        logfile.write('\nTest total losses:\n')
        logfile.write(', '.join([str(item) for item in vlosses]))
        logfile.write('\nTest times:\n')
        logfile.write(', '.join([str(item) for item in vtimes]))
        
        logfile.write('\nBar Precisions:\n')
        logfile.write(', '.join([str(item) for item in barPs]))
        logfile.write('\nBar Recalls:\n')
        logfile.write(', '.join([str(item) for item in barRs]))
        logfile.write('\nBar F1s:\n')
        logfile.write(', '.join([str(item) for item in barF1s]))
        logfile.write('\nTick Precisions:\n')
        logfile.write(', '.join([str(item) for item in tickPs]))
        logfile.write('\nTick Recalls:\n')
        logfile.write(', '.join([str(item) for item in tickRs]))
        logfile.write('\nTick F1s:\n')
        logfile.write(', '.join([str(item) for item in tickF1s]))
        logfile.write('\nUpper Error Precisions:\n')
        logfile.write(', '.join([str(item) for item in euPs]))
        logfile.write('\nUpper Error Recalls:\n')
        logfile.write(', '.join([str(item) for item in euRs]))
        logfile.write('\nUpper Error F1s:\n')
        logfile.write(', '.join([str(item) for item in euF1s]))
        logfile.write('\nLower Error Precisions:\n')
        logfile.write(', '.join([str(item) for item in edPs]))
        logfile.write('\nLower Error Recalls:\n')
        logfile.write(', '.join([str(item) for item in edRs]))
        logfile.write('\nLower Error F1s:\n')
        logfile.write(', '.join([str(item) for item in edF1s]))
        
        logfile.write('\nMean F1s:\n')
        logfile.write(', '.join([str(item) for item in meanF1s]))