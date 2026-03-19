import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from time import time
from tqdm import tqdm
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

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
if device  == "cuda":
    cuda_mem = torch.cuda.get_device_properties(0).total_memory
    print(f"total GPU memory in bytes is \
          \n total: {cuda_mem} \
          \n reserved: {torch.cuda.memory_reserved(0)} \
          \n allocated: {torch.cuda.memory_allocated(0)}"
          )
    print(f"Maximum Batch size: {cuda_mem * batch_size / X_t.nbytes}")
'''
A Transformer is a pipeline of methods applied to the data fed to your
neural net. Transformation is needed to condition your data for proper training.
This can imply a tranformation from integers to floats, and a scaling to
ensure they are in the range [0-1[. If the images are in a so-called
pillow (PIL) format, readable by the Python Image Library, they need to be
transformed to a pytorch Tensor format.
'''
transformer = transforms.Compose(
    [
        transforms.PILToTensor(), # converts PIL to Tensor format
        # transforms data in tensor to 32-bit float and normalizes values to [0, 1]
        # Note the difference if you set scale to False
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

# The batch size is one of the main hyperparameters for the model
batch_size = 64
trainloader = DataLoader(trainset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=16)
valloader = DataLoader(valset,
                       batch_size=batch_size,
                       shuffle=True,
                       num_workers=16)
# Inspect the training set and validation set
ntrain = len(trainset)
nval = len(valset)
print(f"The training set contains {ntrain} samples")
print(f"The validation set contains {nval} samples")
'''
# inspect the data in the training set
X_t, y_t = next(iter(trainloader))
print(f"Dimension of X: {X_t.dim()}")
print(f"Dimension of y: {y_t.dim()}")
print(f"Shape of X [N, C, H, W]: {X_t.shape}")
print(f"Shape of y: {y_t.shape}. Data type {y_t.dtype}")
print(f"Size of an item in X: {X_t.itemsize}")
print(f"Number of elements in X: {X_t.nelement()}")
print(f"Size of X: {X_t.nbytes} = {X_t.nelement()*X_t.itemsize}")
if device  == "cuda":
    print(f"Maximum Batch size: {cuda_mem*batch_size/X_t.nbytes}")

print(X_t)
print(y_t)
print(f"[min-max] value of X: [{X_t.min()}-{X_t.max()}]")
if X_t.dtype == torch.float32:
    print(f"mean and stddev of X: {X_t.mean()}, {X_t.std()}")
# The 2 statements below are demonstration of flattening of a Tensor    
m = nn.Flatten()
flatim = m(X_t)
print(f"The size and length of the flattened input is: {flatim.size()}, {len(flatim)}")
'''
'''
Here we will start to build a simple feed-forward neural net
with 28*28=784 input nodes (to use the grayscale of each pixel
in the 28*28 pixel images of the MNIST dataset as input to the
neural net. The neural net has 784 inputs, corresponding to the
greyscale of each pixel. In order to feed each image of 28*28 pixels
to the input leayer it needs to be transformed to a flattened 1D tensor
of size 784. We use the Flatten() method to do that.
The intermediate layers of the architecture is up to you
to play with. The output layer should consist of 10 nodes that 
represent the number identification from 0-9.

IOUr model below is a Linear model, meaning that all data x is subject to a linear
transformation x.A+b in each of the nodes of each layer.
The weights given to each element of x are contained in the matrix A and bias b.
Troughout the training we will update these weights to minimize a specific 
loss function. This will be taken care of by an optimizer object.

The layers are defined as attributes of the user class Model
that inherits from the superclass nn.Module.
Do not forget to initialize the superclass!
Each user defined model must define a method called forward.
In the forward method you pass the incoming data x through
each of your layers. For more info on linear models see:
https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

After passing data trough a layer of neurons
you need to trigger the neurons with an activation function.
Standard activation functions are contained in the nn.functional
library. The rectified linear unit function, or relu() is 
already implemented in that library.
Experiment yourself with different activation functions, they
can be found here: https://pytorch.org/docs/stable/nn.functional.html
'''

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
        #return F.softmax(x)
        return x

model= MyModel().to(device)
print(model)
'''
Loss function and optimizer:
The loss function is calculated for each data sample that passes through the network
if you pass data in batches through the network, the reduction='mean' option, which is
the default for any loss function, computes the mean of the losses over that given batch
If you set reduction='none', then you will get as output for Loss a Tenor with length
equal to the batch size that represents the loss for each sample in the batch.
You can also use the reduction='sum' option to compute the sum of losses over a given batch.
You can even give weights to certain outputs if you have an unbalanced training sample.
Note that the loss for a single sample is subject to statistical fluctuations, so it is more natural to average
the loss over a sensible batch size. The backward() function will by default calculate the gradients
of the loss function in order to adjust the weights of the network. It can only run by default
when the output of the loss function is a scalar (not a tensor).
The optimizer.step() function will update the weights in order to make one step closer to the
minimum of the loss function. This is typically done after all samples
of a given batch have been passed, and after their mean loss has been computed.
The SGD optimizer will use the gradients of teh loss function wrt. the network weights,
(model parameters) and reduce the parameters with a multiple value (gamma)* this gradient.
the gamma parameter is the learning rate. Large learning rates take big steps towards the
minimum of the loss function, which can oscillate around the true minimum. Small learning rates
can be inefficient and will require lots of epochs to converge, thus increasing the total learning time.
'''
loss_fn = nn.CrossEntropyLoss(reduction='mean')
learning_rate = 0.01
momentum = 0.9
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
print(loss_fn)
print(optimizer)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False,
                        unit="batch", colour="blue")

    for batch, (X, y) in enumerate(progress_bar):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Update progress bar with live loss
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss /= num_batches
    accuracy = correct / size
    return train_loss, accuracy
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    all_preds, all_labels = [], []

    progress_bar = tqdm(dataloader, desc="Validation", leave=False,
                        unit="batch", colour="green")

    with torch.no_grad():
        for X, y in progress_bar:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            predicted = pred.argmax(1)
            correct += (predicted == y).type(torch.float).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # Update progress bar with live loss
            progress_bar.set_postfix(loss=f"{test_loss/num_batches:.4f}")

    test_loss /= num_batches
    accuracy = correct / size
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    return test_loss, accuracy, precision

# important, put the main training loop in the main() program, if you want to use multiprocessing
# History containers
train_losses, val_losses         = [], []
train_accuracies, val_accuracies = [], []
val_precisions                   = []

if __name__ == '__main__':
    epochs = 10
    time0 = time()

    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch", colour="yellow")

    for t in epoch_bar:
        train_loss, train_acc        = train(trainloader, model, loss_fn, optimizer)
        val_loss, val_acc, val_prec  = test(valloader,   model, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        val_precisions.append(val_prec)

        # Update epoch bar with summary metrics
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{100*val_acc:.1f}%",
            val_prec=f"{100*val_prec:.1f}%"
        )

    print("\nDone!")
    print(f"Training Time (in minutes) = {(time()-time0)/60:.2f}")



    epochs_range = range(1, epochs + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    ax1.plot(epochs_range, train_losses, label="Train Loss")
    ax1.plot(epochs_range, val_losses, label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Accuracy
    ax2.plot(epochs_range, train_accuracies, label="Train Accuracy")
    ax2.plot(epochs_range, val_accuracies, label="Val Accuracy")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    # Precision
    ax3.plot(epochs_range, val_precisions, label="Val Precision", color="green")
    ax3.set_title("Precision (macro)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Precision")
    ax3.legend()

    plt.suptitle("MNIST ANN Training Metrics", fontsize=14)
    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=150)  # saves to disk
    plt.show()