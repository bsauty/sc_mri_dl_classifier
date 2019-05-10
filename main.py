from tensorboardX import SummaryWriter
import time
import shutil
import sys
import pickle
import numpy as np
import json
import os
from tqdm import tqdm_notebook

import utils

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
    for bids_ds in tqdm_notebook(context["bids_path_train"], desc="Loading training set"):
        ds_train = BidsDataset(bids_ds,
                               transform=train_transform)
        train_datasets.append(ds_train)

    ds_train = ConcatDataset(train_datasets)
    print(f"Loaded {len(ds_train)} axial slices for the training set.")
    train_loader = DataLoader(ds_train, batch_size=context["batch_size"],
                              shuffle=True, pin_memory=True,
                              collate_fn=mt_datasets.mt_collate,
                              num_workers=1)

    # Validation dataset ------------------------------------------------------
    validation_datasets = []
    for bids_ds in tqdm_notebook(context["bids_path_validation"], desc="Loading validation set"):
        ds_val = BidsDataset(bids_ds,
                             transform=val_transform)
        validation_datasets.append(ds_val)

    ds_val = ConcatDataset(validation_datasets)
    print(f"Loaded {len(ds_val)} axial slices for the validation set.")
    val_loader = DataLoader(ds_val, batch_size=context["batch_size"],
                            shuffle=True, pin_memory=True,
                            collate_fn=mt_datasets.mt_collate,
                            num_workers=1)

    model = Classifier(drop_rate=context["dropout_rate"],
                       bn_momentum=context["batch_norm_momentum"])
    model.cuda()

    num_epochs = context["num_epochs"]
    initial_lr = context["initial_lr"]

    # Using Adam with cosine annealing learning rate
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # Write the metrics, images, etc to TensorBoard format
    writer = SummaryWriter(log_dir=context["log_directory"])

    # Training loop -----------------------------------------------------------
    best_validation_loss = float("inf")
    for epoch in tqdm_notebook(range(1, num_epochs+1), desc="Training"):
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
            var_labels = OneHotEncode(input_labels).cuda()

            preds = model(var_input)

            CE_loss = nn.BCELoss()
            loss = CE_loss(preds, var_labels)
            train_loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            num_steps += 1

        train_loss_total_avg = train_loss_total / num_steps

        #tqdm.write(f"Epoch {epoch} training loss: {train_loss_total_avg:.4f}.")
        print(f"Epoch {epoch} training loss: {train_loss_total_avg:.4f}.")
        
        # Validation loop -----------------------------------------------------
        model.eval()
        val_loss_total = 0.0
        num_steps = 0

        for i, batch in enumerate(val_loader):
            input_samples = batch["input"]
            input_labels = get_modality(batch)
            
            with torch.no_grad():
                var_input = input_samples.cuda()
                var_labels = OneHotEncode(input_labels).cuda()

                preds = model(var_input)

                CE_loss = nn.BCELoss()
                loss = CE_loss(preds, var_labels)
                val_loss_total += loss.item()

            num_steps += 1

        val_loss_total_avg = val_loss_total / num_steps

        #tqdm.write(f"Epoch {epoch} validation loss: {val_loss_total_avg:.4f}.")
        print(f"Epoch {epoch} validation loss: {val_loss_total_avg:.4f}.")
        
        end_time = time.time()
        total_time = end_time - start_time
        #tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))
        print("Epoch {} took {:.2f} seconds.".format(epoch, total_time))
        
        if val_loss_total_avg < best_validation_loss:
            best_validation_loss = val_loss_total_avg
            torch.save(model, "./"+context["log_directory"]+"/best_model.pt")

    # save final model
    torch.save(model, "./"+context["log_directory"]+"/final_model.pt")
    
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
