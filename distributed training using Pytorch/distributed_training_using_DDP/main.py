import os

import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from model.SimpleModelClass import SimpleNetModel
from model.DefaultModelTraining import default_training_step


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


def training_step_DDP(rank, world_size):
    print("starts training step for device rank: ", rank)

    setup_ddp(rank, world_size)

    model = SimpleNetModel().to(rank)

    model_ddp = DDP(model, device_ids=[rank])

    optimizer = optim.SGD(model_ddp.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    optimizer.zero_grad()

    predictions = model_ddp(torch.randn(20, 10))
    labels = torch.randn(20, 1).to(rank)

    losses = loss_fn(labels, predictions)
    losses.backward()

    print("losses : ", losses)
    optimizer.step()

    clear_process()


def run_step(model_fn, world_size):
    """
    :param model_fn: function that needs to be called in a distributed manner
    :param world_size: total number of processes
    """
    print("run_step called, world_size: ", world_size)
    mp.spawn(model_fn, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    print(" DDP Program starts ")

    device, world_size = verify_gpu_devices()

    if device.type == 'cuda' and world_size >= 2:
        print(" System Setup satisfies the single node & multi-processes training design")
        run_step(training_step_DDP, world_size)
    elif device.type == 'cpu':
        print(" System setup does not satisfies for DDP training, Use default training process")
        default_training_step()
    else:
        print(" other cases needs to be handled differently")


    # Above script needs to be run simply with terminal:
    # -> python3 main.py
