from tensorboardX import SummaryWriter
import time
import shutil
import sys
import json
import numpy as np
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

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from PIL import Image
import matplotlib.pyplot as plt

import os

from classifier import loader
from classifier import model as M
from classifier import utils

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


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
    writer = SummaryWriter(logdir=context["log_directory"])
    
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
            input_labels = utils.get_modality(batch)
            
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
        
        #setting the lists for confusion matrix
        true_labels = []
        guessed_labels = []


        for i, batch in enumerate(val_loader):
            input_samples = batch["input"]
            input_labels = utils.get_modality(batch)

            true_labels += input_labels
            
            with torch.no_grad():
                var_input = input_samples.cuda()
                var_labels = torch.cuda.LongTensor(input_labels).cuda(non_blocking=True)

                outputs = model(var_input)
                _, preds = torch.max(outputs, 1)
                
                lst_labels = [int(x) for x in preds]
                guessed_labels += lst_labels

                loss = criterion(outputs, var_labels)
                val_loss_total += loss.item()


            num_steps += 1

            
        accuracy = accuracy_score(true_labels, guessed_labels)
        recall = recall_score(true_labels, guessed_labels, average=None)
        precision = precision_score(true_labels, guessed_labels, average=None)
            
            
        val_loss_total_avg = val_loss_total / num_steps
        lst_val_loss.append(val_loss_total_avg)
        tqdm.write(f"Epoch {epoch} validation loss: {val_loss_total_avg:.4f}.")
        
        lst_accuracy.append(accuracy)
        tqdm.write(f"Epoch {epoch} accuracy : {accuracy:.4f}.")
        
        # add metrics for tensorboard
        writer.add_scalars('validation metrics', {
            'accuracy' :accuracy,
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
            torch.save(model.state_dict(), "./"+context["log_directory"]+"/best_model.pt")

    # save final model
    torch.save(model.state_dict(), "./"+context["log_directory"]+"/final_model.pt")
    
    
    return

def cmd_test(context):

    # Set the GPU
    gpu_number = context["gpu"]
    torch.cuda.set_device(gpu_number)

    # These are the validation/testing transformations
    val_transform = transforms.Compose([
        mt_transforms.CenterCrop2D((128, 128)),
        mt_transforms.ToTensor(),
        mt_transforms.NormalizeInstance(),
    ])

    test_datasets = []
    for bids_ds in tqdm(context["bids_path_test"], desc="Loading test set"):
        ds_test = loader.BidsDataset(bids_ds,
                                     transform=val_transform,
                                     slice_filter_fn=loader.SliceFilter())
        test_datasets.append(ds_test)


    ds_test = ConcatDataset(test_datasets)
    tqdm.write(f"Loaded {len(ds_test)} axial slices for the test set.")
    test_loader = DataLoader(ds_test, batch_size=context["batch_size"],
                             shuffle=True, pin_memory=True,
                             collate_fn=mt_datasets.mt_collate,
                             num_workers=1)

    model = M.Classifier()
    model.load_state_dict(torch.load("./"+context["log_directory"]+"/best_model.pt", map_location="cuda:0"))
    model.cuda()
    model.eval()
    
    #setting the lists for confusion matrix
    true_labels = []
    guessed_labels = []

    for i, batch in enumerate(test_loader):
        input_samples = batch["input"]
        input_labels = utils.get_modality(batch)
        
        true_labels += input_labels

        with torch.no_grad():
            test_input = input_samples.cuda()
            test_labels = torch.cuda.LongTensor(input_labels).cuda(non_blocking=True)

            outputs = model(test_input)
            _, preds = torch.max(outputs, 1)
            
            lst_labels = [int(x) for x in preds]
            guessed_labels += lst_labels
                    
    accuracy = accuracy_score(true_labels, guessed_labels)
    recall = recall_score(true_labels, guessed_labels, average=None)
    precision = precision_score(true_labels, guessed_labels, average=None)
    
    '''
    np.set_printoptions(precision=2)
    
    if not(os.path.exists("./temp/")):
        os.makedirs("./temp/")
        
    class_names = ["MToff_MTS", "MTon_MTS", "T1w_MTS", "T1w", "T2star", "T2w"]
    # Plot normalized confusion matrix
    utils.plot_confusion_matrix(true_labels, guessed_labels, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig("./temp/test_cm.png")
    utils.plot_metrics(np.array([recall, precision]), accuracy, class_names)
    plt.savefig("./temp/test_accuracy.png")
    
    open("./temp/test_cm.png")
    open("./temp/test_accuracy.png")'''
    
    tqdm.write(f"Accuracy over test slices : {accuracy}")
    tqdm.write(f"Recall over test slices : {recall}")
    tqdm.write(f"Precision over test slices : {precision}")
    

    return


def run_main():
    if len(sys.argv) <= 1:
        print("\nclassify [config.json]\n")
        return

    with open(sys.argv[1], "r") as fhandle:
        context = json.load(fhandle)

    command = context["command"]

    if command == 'train':
        cmd_train(context)
        shutil.copyfile(sys.argv[1], "./"+context["log_directory"]+"/config_file.json")
    elif command == 'test':
        cmd_test(context)

run_main()
