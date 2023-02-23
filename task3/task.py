import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import resnet50
from torch.utils.data.dataset import random_split
from task2.task import MixUp
from time import time
from typing import Tuple

"""
Note:

1) Training accuracy is a running average over the epoch.
This means that the model changes while the average is taken 
and it is a biased estimate. 

2) The training data has the MixUp transform applied. In 
contrast the validation set does not.

Both of these factors lead to the training loss and the 
validation loss not being comparable when printed.
"""
def opt_ablation(opt: optim.Optimizer, net: nn.Module)-> Tuple[float, float]:
    """Trains Network with a specific optimizer.

    Returns the final validation loss and accuracy of the model.
    """
    criterion = torch.nn.CrossEntropyLoss()
    print("------------------------------")
    ## train
    for epoch in range(15):

        running_acc = 0.0
        running_loss = 0.0

        for i, train_data in enumerate(train_loader):

            inputs, labels = train_data
            inputs = inputs.to(device)
            labels = labels.squeeze().to(device)

            opt.zero_grad()
            outputs = net(inputs)

            # Accuracy
            output_class = torch.argmax(outputs, dim=1)
            label_class = torch.argmax(labels, dim=1)
            accuracy = (output_class == label_class).float().sum()

            # BCE
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            running_loss += loss.item() * labels.shape[0]
            running_acc += accuracy.item()

        for val_batch in val_loader:
            with torch.no_grad():
                val_input, val_labels = val_batch
                val_input = val_input.to(device)
                val_labels = val_labels.squeeze().to(device)

                val_out = net(val_input)
                val_out_class = torch.argmax(val_out, dim=1)
                val_acc = (val_out_class == val_labels).float().mean().item()

                val_loss = criterion(val_out, val_labels).item()

        # Perform mean over batches
        running_acc = running_acc / len(train_set)
        running_loss = running_loss / len(train_set)

        print(f"Epoch {epoch + 1}:")
        print(f"Train Loss: {running_loss:.2e}")
        print(f"Train Accuracy: {running_acc * 100:.0f}%")
        print(f"Validation Loss: {val_loss:.2e}")
        print(f"Validation Accuracy: {val_acc * 100:.0f}%")
        print("------------------------------")
    return val_loss, val_acc


if __name__ == "__main__":
    print("Ablation Study on differing Optimizers.")
    print("Training Data augmented with MixUp lam ~ beta(0.2, 0.2)")
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 256
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    #
    data_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_val_set, test_set = random_split(data_set, [0.8, 0.2])
    train_set, val_set = random_split(train_val_set, [0.9, 0.1])


    train_loader = DataLoader(
        MixUp(train_set, len(classes), "beta"), batch_size, num_workers=4
    )
    val_loader = DataLoader(val_set, batch_size=len(val_set), num_workers=4)
    test_loader = DataLoader(test_set, batch_size=len(test_set), num_workers=8)


    print("------------------------------")
    print("-------------SGD--------------")
    print("lr = 0.01, momentum = 0.9")

    sgd_net = resnet50()
    sgd_net.fc = nn.Linear(2048, len(classes)) # rehead for 10 class rather than 1000
    sgd_net.to(device)

    sgd = optim.SGD(sgd_net.parameters(), lr=0.01, momentum=0.9)

    t0_sgd = time()
    sgd_val_loss, sgd_val_acc = opt_ablation(sgd, sgd_net)
    t1_sgd = time()

    torch.save(sgd_net.state_dict(), "sgd_model.pt")
    print("SGD Model saved.")

    print("------------------------------")
    print("-------------ADAM-------------")
    print("Default hyperparameters.")

    adam_net = resnet50()
    adam_net.fc = nn.Linear(2048, len(classes))
    adam_net.to(device)

    opt = optim.Adam(adam_net.parameters())

    t0_adam = time()
    adam_val_loss, adam_val_acc = opt_ablation(opt, adam_net)
    t1_adam = time()

    print("------------------------------")

    torch.save(adam_net.state_dict(), "adam_model.pt")
    print("Adam Model saved.")

    print("-----------Results------------")

    loss = nn.CrossEntropyLoss()
    for test_batch in test_loader:
        with torch.no_grad():

            test_in, test_lab = test_batch
            test_in = test_in.to(device)
            test_lab = test_lab.to(device)

            adam_pred = adam_net(test_in)
            adam_class = torch.argmax(adam_pred, dim=1)
            adam_acc = (adam_class == test_lab).float().mean().item()
            adam_loss = loss(adam_pred, test_lab).item()

            sgd_pred = sgd_net(test_in)
            sgd_class = torch.argmax(sgd_pred, dim=1)
            sgd_acc = (sgd_class == test_lab).float().mean().item()
            sgd_loss = loss(sgd_pred, test_lab).item()

    print("Adam:")
    print(f"Traing Time: {t1_adam-t0_adam:.2f}s")
    print(f"Test Accuracy: {adam_acc*100:.2f}%")
    print(f"Test Loss: {adam_loss:.2f}")
    print(f"Validation Accuracy: {adam_val_acc*100:.2f}%")
    print(f"Validation Loss: {adam_val_loss:.2f}")
    print("--------------------------------")
    print("SGD:")
    print(f"Traing Time: {t1_sgd-t0_sgd:.2f}s")
    print(f"Test Accuracy: {sgd_acc*100:.2f}%")
    print(f"Test Loss: {sgd_loss:.2f}")
    print(f"Validation Accuracy: {sgd_val_acc*100:.2f}%")
    print(f"Validation Loss: {sgd_val_loss:.2f}")


