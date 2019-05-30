from tensorboardX import SummaryWriter
import time
import shutil
import sys
import json
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, ConcatDataset


from bids_neuropoly import bids
from medicaltorch import datasets as mt_datasets
from medicaltorch import transforms as mt_transforms
from medicaltorch import filters as mt_filters

from PIL import Image

import os

import loader
import model as M

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def get_modality(batch):
    labels = []
    for acq in batch['input_metadata']:
        path = acq.__getitem__('bids_metadata')['input_filename']
        name = os.path.basename(path)
        if "acq-MToff_MTS" in name :
            labels.append(0)
            continue
        if "acq-MTon_MTS" in name :
            labels.append(1)
            continue
        if "acq-T1w_MTS" in name :
            labels.append(2)
            continue
        if "T1w" in name :
            labels.append(3)
            continue
        if "T2star" in name :
            labels.append(4)
            continue
        if "T2w" in name :
            labels.append(5)
            continue
    return labels

def OneHotEncode(labels):
    ohe_labels = []
    for label in labels :
        ohe = [0 for i in range(6)]
        ohe[label] = 1
        ohe_labels.append(ohe)
    return torch.FloatTensor(ohe_labels)

def cmd_train(context):
    """Main command do train the network.
    :param context: this is a dictionary with all data from the
                    configuration file:
                        - 'command': run the specified command (e.g. train, test)
                        - 'gpu': ID of the used GPU
                        - 'bids_path_train': list of relative paths of the BIDS folders of each training center
                        - 'bids_path_validation': list of relative paths of the BIDS folders of each validation center
                        - 'bids_path_test': list of relative paths of the BIDS folders of each test center
                        - 'batch_size'
                        - 'dropout_rate'
                        - 'batch_norm_momentum'
                        - 'num_epochs'
                        - 'initial_lr': initial learning rate
                        - 'log_directory': folder name where log files are saved
    """
    # Set the GPU
    gpu_number = context["gpu"]
    torch.cuda.set_device(gpu_number)

    # These are the training transformations
    train_transform = transforms.Compose([
        mt_transforms.CenterCrop2D((128, 128)),
        mt_transforms.ElasticTransform(alpha_range=(28.0, 30.0),
                                       sigma_range=(3.5, 4.0),
                                       p=0.3),
        mt_transforms.RandomAffine(degrees=4.6,
                                   scale=(0.98, 1.02),
                                   translate=(0.03, 0.03)),
        mt_transforms.RandomTensorChannelShift((-0.10, 0.10)),
        mt_transforms.ToTensor(),
        mt_transforms.NormalizeInstance(),
    ])

    # These are the validation/testing transformations
    val_transform = transforms.Compose([
        mt_transforms.CenterCrop2D((128, 128)),
        mt_transforms.ToTensor(),
        mt_transforms.NormalizeInstance(),
    ])

    # This code will iterate over the folders and load the data, filtering
    # the slices without labels and then concatenating all the datasets together
    train_datasets = []
    for bids_ds in tqdm(context["bids_path_train"], desc="Loading training set"):
        ds_train = loader.BidsDataset(bids_ds,
                               transform=train_transform,
                               slice_filter_fn=loader.SliceFilter())
        train_datasets.append(ds_train)

    ds_train = ConcatDataset(train_datasets)
    print(f"Loaded {len(ds_train)} axial slices for the training set.")
    train_loader = DataLoader(ds_train, batch_size=context["batch_size"],
                              shuffle=True, pin_memory=True,
                              collate_fn=mt_datasets.mt_collate,
                              num_workers=1)

    # Validation dataset ------------------------------------------------------
    validation_datasets = []
    for bids_ds in tqdm(context["bids_path_validation"], desc="Loading validation set"):
        ds_val = loader.BidsDataset(bids_ds,
                             transform=val_transform,
                             slice_filter_fn=loader.SliceFilter())
        validation_datasets.append(ds_val)

    ds_val = ConcatDataset(validation_datasets)
    print(f"Loaded {len(ds_val)} axial slices for the validation set.")
    val_loader = DataLoader(ds_val, batch_size=context["batch_size"],
                            shuffle=True, pin_memory=True,
                            collate_fn=mt_datasets.mt_collate,
                            num_workers=1)

    model = M.Classifier(drop_rate=context["dropout_rate"],
                       bn_momentum=context["batch_norm_momentum"])
    model.cuda()

    num_epochs = context["num_epochs"]
    initial_lr = context["initial_lr"]

    # Using SGD with cosine annealing learning rate
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # Write the metrics, images, etc to TensorBoard format
    writer = SummaryWriter(log_dir=context["log_directory"])
    
    # Cross Entropy Loss
    criterion = nn.CrossEntropyLoss()
    
    # Training loop -----------------------------------------------------------
    best_validation_loss = float("inf")
    
    lst_train_loss = []
    lst_val_loss = []
    lst_accuracy = []

    for epoch in tqdm(range(1, num_epochs+1), desc="Training"):
        start_time = time.time()

        scheduler.step()

        lr = scheduler.get_lr()[0]
        writer.add_scalar('learning_rate', lr, epoch)

        model.train()
        train_loss_total = 0.0
        num_steps = 0
        
        for i, batch in enumerate(train_loader):
            input_samples = batch["input"]
            input_labels = get_modality(batch)
            
            var_input = input_samples.cuda()
            var_labels = torch.cuda.LongTensor(input_labels).cuda(non_blocking=True)

            outputs = model(var_input)

            loss = criterion(outputs, var_labels)
            train_loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            num_steps += 1

        train_loss_total_avg = train_loss_total / num_steps
        lst_train_loss.append(train_loss_total_avg)

        tqdm.write(f"Epoch {epoch} training loss: {train_loss_total_avg:.4f}.")
        
        # Validation loop -----------------------------------------------------
        model.eval()
        val_loss_total = 0.0
        num_steps = 0
        
        val_accuracy = 0
        num_samples = 0


        for i, batch in enumerate(val_loader):
            input_samples = batch["input"]
            input_labels = get_modality(batch)
            
            with torch.no_grad():
                var_input = input_samples.cuda()
                var_labels = torch.cuda.LongTensor(input_labels).cuda(non_blocking=True)

                outputs = model(var_input)
                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, var_labels)
                val_loss_total += loss.item()

                val_accuracy += int((var_labels == preds).sum())

            num_steps += 1
            num_samples += context['batch_size']

            
        val_loss_total_avg = val_loss_total / num_steps
        lst_val_loss.append(val_loss_total_avg)
        tqdm.write(f"Epoch {epoch} validation loss: {val_loss_total_avg:.4f}.")
        
        val_accuracy_avg = 100 * val_accuracy / num_samples
        lst_accuracy.append(val_accuracy_avg)
        tqdm.write(f"Epoch {epoch} accuracy : {val_accuracy_avg:.4f}.")
        
        # add metrics for tensorboard
        writer.add_scalars('validation metrics', {
            'accuracy' :val_accuracy_avg,
        }, epoch)
        
        writer.add_scalars('losses', {
            'train_loss': train_loss_total_avg,
            'val_loss': val_loss_total_avg,
        }, epoch)

        
        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))
        
        if val_loss_total_avg < best_validation_loss:
            best_validation_loss = val_loss_total_avg
            torch.save(model, "./"+context["log_directory"]+"/best_model.pt")

    # save final model
    torch.save(model, "./"+context["log_directory"]+"/final_model.pt")
    
    # save the metrics
    parameters = "CrossEntropyLoss/batchsize=" + str(context['batch_size'])
    parameters += "/initial_lr=" + str(context['initial_lr'])
    parameters += "/dropout=" + str(context['dropout_rate'])
        
    plt.subplot(2,1,1)
    plt.title(parameters)
    plt.plot(lst_train_loss, color='red', label='Training')
    plt.plot(lst_val_loss, color='blue', label='Validation')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    
    plt.subplot(2,1,2)
    plt.plot(lst_accuracy)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    
    plt.savefig(parameters+'.png')
    
    return


def run_main():
    if len(sys.argv) <= 1:
        print("\nivadomed [config.json]\n")
        return

    with open(sys.argv[1], "r") as fhandle:
        context = json.load(fhandle)

    command = context["command"]

    if command == 'train':
        cmd_train(context)
        shutil.copyfile(sys.argv[1], "./"+context["log_directory"]+"/config_file.json")
    elif command == 'test':
        cmd_test(context)

        
if __name__ == "__main__":
    run_main()
