import torch
import torch.nn as nn

# to monitor your NVIDIA GPU permanently type in a window
# watch -n 3 nvidia-smi --query-gpu=index,gpu_name,memory.total,memory.used,memory.free,temperature.gpu,pstate,utilization.gpu,utilization.memory --format=csv
# setting device on GPU if available, else CPU
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', dev)

#Additional Info when using cuda
if dev.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


criterion = nn.NLLLoss()

images, labels = next(iter(trainloader))
images.to(device)
labels.to(device)
images = images.view(images.shape[0], -1)



logps = model(images) #log probabilities



time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        images.to(device)
        labels.to(device)
        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))
        print("\nTraining Time (in minutes) =", (time() - time0) / 60)

