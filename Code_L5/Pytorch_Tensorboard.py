import torch
import torchvision.utils
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow as matplotlib_imshow
import numpy as np
from time import time

# Print the pytorch version number
print(f"Using Pytorch version {torch.__version__}")
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
transformer = transforms.Compose(
    [
        transforms.PILToTensor(),
        transforms.ToDtype(dtype=torch.float32, scale=True),
    ]
)
# Download and load the training data
trainset = datasets.MNIST(
    root="data", download=True, train=True,
    transform=transformer)
valset = datasets.MNIST(
    root="data", download=True, train=False,
    transform=transformer)

# Inspect the training set and validation set
ntrain = len(trainset)
nval = len(valset)
print(f"The training set contains {ntrain} samples")
print(f"The validation set contains {nval} samples")

# layer parameters
input_size = 28*28
hidden_sizes = [128, 64]
output_size = 10
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_sizes[0])
        self.fc2 = nn.Linear(in_features=hidden_sizes[0], out_features=hidden_sizes[1])
        self.fc3 = nn.Linear(in_features=hidden_sizes[1], out_features=output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Scan over lists of Model params
batch_sizes = [2, 64, 1024]
learning_rates = [0.1, 0.01, 0.001, 0.0001]


for batch_size in batch_sizes:
    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)
    valloader = DataLoader(valset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=4)
    for learning_rate in learning_rates:

        model = MyModel().to(device)
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # make a tensorboard summary writer
        train_logdir = f'runs/MNIST_experiment2/train/BS {batch_size} LR {learning_rate}'
        train_writer = SummaryWriter(log_dir=train_logdir)
        val_logdir = f'runs/MNIST_experiment2/val/BS {batch_size} LR {learning_rate}'
        val_writer = SummaryWriter(log_dir=val_logdir)

        # initialize the SummaryWriter step
        step = 0
        train_size = len(trainloader.dataset)
        val_size = len(valloader.dataset)
        num_val_batches = len(valloader)
        time0 = time()
        #Start training and validation
        epochs = 50
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            model.train()
            # initialize accuracy measure
            val_loss, val_correct, train_correct = 0, 0, 0
            for batch, (X, y) in enumerate(trainloader):
                # Zero the gradients for each batch
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                # Compute prediction error
                pred = model(X)
                loss = loss_fn(pred, y)
                # Backpropagation
                loss.backward()
                optimizer.step()
                # calculate the running training accuracy
                predictions = pred.argmax(1)
                train_correct += (predictions == y).sum()
                # add images to Tensorboard
                img_grid = torchvision.utils.make_grid(X)
                train_writer.add_image('minst_images', img_grid)
                train_writer.add_histogram('fc1', model.fc1.weight)
                train_writer.add_histogram('fc2', model.fc2.weight)
                train_writer.add_histogram('fc3', model.fc3.weight)
                #train_writer.add_embedding()
            train_loss = loss
            train_acc = float(train_correct) / train_size
            print(f"Train loss: {train_loss:>7f}, Train accuracy: {train_acc:>4f}, \
                    Train size: {train_size:>5d}]")
            # switching to validation
            model.eval()
            # print("testing over whole test set")
            with torch.no_grad():
                for X, y in valloader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)
                    val_loss += loss_fn(pred, y).item()
                    # print(f"index of maximum value of predictions {pred.argmax(1)}")
                    val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            val_loss /= num_val_batches
            val_acc = val_correct / val_size
            print(f"Validation loss: {val_loss:>7f}, Validation accuracy: {val_acc:>4f}, \
                    Validation size: {val_size:>5d}]")
            # Write results to tensorboard
            train_writer.add_scalar('Loss', train_loss, global_step=step)
            train_writer.add_scalar('Accuracy', train_acc, global_step=step)
            val_writer.add_scalar('Loss', val_loss, global_step=step)
            val_writer.add_scalar('Accuracy', val_acc, global_step=step)
            train_writer.add_hparams({'lr': learning_rate,'bsize': batch_size},
                                   {'TrainLoss': train_loss, 'TrainAcc': train_acc})
            val_writer.add_hparams({'lr': learning_rate, 'bsize': batch_size},
                                     {'ValLoss': val_loss, 'ValAcc': val_acc})

            step += 1
        print(f"Done scan: \n batch size: {batch_size}, learning rate: {learning_rate}")
        print("\nTraining Time (in minutes) =", (time() - time0) / 60)
        train_writer.flush()
        val_writer.flush()
print("All Done!")
