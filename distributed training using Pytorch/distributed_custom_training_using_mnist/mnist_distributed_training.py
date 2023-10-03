import os

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.optim as optim
from model.SimpleImageClassificationModel import ConvolutionalNNModel
from data.MNISTDataDownloader import getTrainDataset
from utils import HelperFunctions


def verify_gpu_devices():
    print("torch version: ", torch.__version__)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    world_size = torch.cuda.device_count()

    print("number of cuda devices: ", world_size)
    print(" Is distributed training available: ", torch.distributed.is_available())

    return device, world_size


def setup_ddp(rank, world_size):
    """
    :param rank: unique identifier for each process
    :param world_size: total number of processes
    """
    print("setup_ddp starts for rank: ", rank)
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '12355'

    # check if NCCL backend for process group communication is available or not
    print("Is NCCL available: ", torch.distributed.is_nccl_available())

    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)


def clear_process():
    print(" Clear all process group after training is done")
    torch.distributed.destroy_process_group()


def training_step_per_batch(model, optimizer, inputs, labels, loss_fn):
    optimizer.zero_grad()

    predictions = model(inputs)

    losses = loss_fn(predictions, labels)

    losses.backward()

    optimizer.step()

    return losses


def get_data_loader(dataset, batch_size):
    # DistributedSampler will sample dataset and assign exclusive parts of dataset
    # to all the workers. No need to do shuffle while using Sampler.
    return DataLoader(dataset,
                      batch_size=batch_size,
                      pin_memory=True,
                      shuffle=True,
                      sampler=DistributedSampler(dataset))


def save_checkpoints(model, epoch):
    chkp = model.module.state_dict()
    path = "./output/checkpoint.pt"
    torch.save(chkp, path)
    print("Training checkpoint saved at step : ", epoch)


def training_step_DDP(rank, world_size, img_size, batch_size, max_epochs, saving_step):
    print("starts training step for device rank: ", rank)

    setup_ddp(rank, world_size)

    model = ConvolutionalNNModel().to(rank)
    model_ddp = DDP(model, device_ids=[rank])

    train_dataset = getTrainDataset(img_size)
    dataloader = get_data_loader(train_dataset, batch_size)

    optimizer = optim.SGD(model_ddp.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    losses_list = []
    for epoch in range(max_epochs):
        print(f"GPU with rank: {rank} and epoch : {epoch} with batch_size: {batch_size}")
        # very important step *******************
        dataloader.sampler.set_epoch(epoch)

        for images, labels in dataloader:
            images = images.to(rank)
            labels = labels.to(rank)

            batch_loss = training_step_per_batch(model_ddp, optimizer, images, labels, loss_fn)
            losses_list.append(batch_loss)

        # Save Checkpoints at every saving_step
        if rank == 0 and (epoch + 1) % saving_step == 0:
            save_checkpoints(model_ddp, epoch)

    print("Accumulated losses list size: ", len(losses_list))
    HelperFunctions.save_loss_graph(losses_list)

    clear_process()


def run_step(model_fn, world_size, img_size, batch_size, max_epochs, save_step):
    """
    :param model_fn: function that needs to be called in a distributed manner
    :param world_size: total number of processes
    """
    print("run_step called, world_size: ", world_size)
    mp.spawn(model_fn, args=(world_size, img_size, batch_size, max_epochs, save_step), nprocs=world_size, join=True)


if __name__ == '__main__':
    print(" DDP Program starts ")

    device, world_size = verify_gpu_devices()
    img_size = 16
    batch_size = 32
    max_epochs = 10
    saving_step = 5

    if device.type == 'cuda' and world_size >= 2:
        print(" System Setup satisfies the single node & multi-processes training design")
        run_step(training_step_DDP, world_size, img_size, batch_size, max_epochs, saving_step)
    else:
        print(" other cases needs to be handled differently")

    # Above script needs to be run simply with terminal:
    # First install required Libraries:
    # ->  torch, torchvision, matplotlib
    # Now, run below script for DDP training on multiple GPUs
    # -> python3 mnist_distributed_training.py
