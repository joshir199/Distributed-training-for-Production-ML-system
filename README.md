# Distributed-training-for-Production-ML-system
Understanding and Implementing various types of distributed training pipelines for Production ML Systems. Training time plays very important role when the model is getting deployed in the production pipeline because production ML system will be very large and will get larger with time and needs to be updated regularly to improve or adapt of changes in dataset. 

Since, multiple GPUs setup are now easily available with the Cloud ecosystem or On-Premises setup, It's always a good idea to make use of those precious resources to make ML system faster. Thus, Using distributed training setup is the best way to work with ML system nowadays. 

There are various types of training setup:
1. CPU with single machine
2. Single GPU with single machine
3. Multi-GPU with single machine
4. Multi-GPU with multi machine
5. TPU setup

When multiple GPUs (either on single machine or multiple machine) are used for training of the ML model, Distributed training setup is needed to executed it without any issues.

# Using Tensorflow
Tensorflow handles distributed training using various Strategies like below:
1. Mirrored Strategy
2. Multi-Mirrored Strategy
3. TPU Strategy
4. Parameter Serving Strategy

For more detailed explanation and concepts behind the distributed architecture, please refer to code in the repo.

And Mirrored Strategy is seen as most widely used setup with multiple GPU on a single machine for training Large models like LLMs, Generative models etc.

# loss calculation in distributed training setup

![](https://github.com/joshir199/Distributed-training-for-Production-ML-system/blob/main/distributed%20training%20using%20tensorflow/distributed_custom_training/Loss_calculation_for_distributed_custom_training.png)


---------------------
# training time graphs for different strategy of training

![](https://github.com/joshir199/Distributed-training-for-Production-ML-system/blob/main/distributed%20training%20using%20tensorflow/distributed_custom_training/training_time_graph.png)

***********************************************
# Using PyTorch

There are a few ways you can perform distributed training in PyTorch with each method having their advantages in certain use cases:
1. Distributed Data Parallel (DDP)
2. Fully Sharded Data Parallel (FSDP)
3. Remote Procedure Call (RPC) distributed training

Out of those, DDP are the most widely used distributed training setup used in Pytorch ML model training. It can deal with various hardware set like, single node-single process,
single node - multi process, multi node - multi process etc.

Main concept of DDP design relies on the communication between the processes while training the ML model.

# All-Reduce collective communication among processes
![](https://github.com/joshir199/Distributed-training-for-Production-ML-system/blob/main/distributed%20training%20using%20Pytorch/collective_communications.png)

For more detailed explanation with example, refer to the code in this repo.

