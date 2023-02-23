import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch import nn
from torch import distributions as dist
from torch.nn import functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import resnet50
from typing import Iterator, Tuple


parser = argparse.ArgumentParser()

parser.add_argument(
    "-sm",
    "--sampling_method",
    help="Defines which distribuition MixUp parameter lambda is drawn from 1: beta 2:Uniform",
    type=int,
    default=1,
)

args = parser.parse_args()
# mixup
class MixUp(Dataset):
    """Implements mixup algorithm on a dataset."""

    def __init__(
        self, origional_dataset: Dataset, classes: int, dist_class: str, alpha=0.2
    ):
        self.classes = classes
        self.data = origional_dataset
        if dist_class == "beta":
            self.dist = dist.Beta(alpha, alpha)
        elif dist_class == "uniform":
            self.dist = dist.Uniform(0, 1)
        else:
            raise ValueError("dist_class must be 'uniform' or 'beta'")

    def __len__(self)-> int:
        return len(self.data)

    def __getitem__(self, idx:int)-> Tuple[torch.Tensor, torch.Tensor]:
        lmbda = torch.clamp(self.dist.sample((1,)), 0, 1)
        j = torch.randint(0, self.__len__(), (1,))
        x_i, y_i = self.data[idx]
        x_j, y_j = self.data[j]
        y_i = F.one_hot(torch.LongTensor([y_i]), num_classes=self.classes)
        y_j = F.one_hot(torch.LongTensor([y_j]), num_classes=self.classes)
        x = x_i * lmbda + (1 - lmbda) * x_j
        y = y_i * lmbda + (1 - lmbda) * y_j
        return x, y


def img_grid(batch_data: Iterator, file_name: str, grid_size: int=4):
    """Produces a n x n grid of images saved as filename."""
    images, _ = next(batch_data)
    images = images[:grid_size**2]

    rows = []
    for start in range(0, grid_size**2, grid_size):
        end = start + grid_size
        row = torch.cat([images[i] for i in range(start, end)], dim=1)
        rows.append(row)

    image_grid = torch.cat(rows, dim=2) * 0.5 * 255 + 0.5 * 255 # stack rows ontop of each other and invert standard transform
    image_grid = image_grid.permute(1, 2, 0).numpy().astype("uint8") # (C, H, W) -> (H, W, C)
    im = Image.fromarray(image_grid)
    im.save(file_name)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Collect input flag
    sampling_method = args.sampling_method

    if sampling_method == 1:
        print("Using Beta Distribution for MixUp")
        sampling = "beta"
    if sampling_method == 2:
        print("Using Uniform Distribution for MixUp")
        sampling = "uniform"
    else:
        ValueError("sampling_method flag must be 1 or 2.")
    print("------------------------------")

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
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    mix_up = MixUp(trainset, len(classes), sampling)
    trainloader = DataLoader(mix_up, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(trainset, batch_size=36)
    # Sample a batch and make an image
    dataiter = iter(trainloader)
    img_grid(dataiter, "./mixup.png")
    print("mixup.png saved.")

    net = resnet50()
    net.fc = nn.Linear(2048, len(classes))  # Rehead resnet50 for 10 class class.

    net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    print("------------------------------")

    for epoch in range(10):
        running_acc = 0.0
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.squeeze()
            labels = labels.to(device)

            outputs = net(inputs)

            output_class = torch.argmax(outputs, dim=1)
            label_class = torch.argmax(labels, dim=1)
            accuracy = (output_class == label_class).float().sum()

            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.shape[0] # add sum to running total.
            running_acc += accuracy.item()

        running_acc = running_acc / len(trainset)
        running_loss = running_loss / len(trainset)

        print(f"Epoch {epoch + 1}:")
        print(f"Loss: {running_loss:.2f}")
        print(f"Accuracy: {running_acc*100:.0f}%")
        print("------------------------------")

    print("Training done.")

    # save trained model
    torch.save(net.state_dict(), f"saved_model_{sampling}.pt")
    print("Model saved.")
    net.to("cpu")

    # import matplotlib.pyplot as plt
    # test_img, test_labels = next(iter(test_loader))
    # test_class = torch.argmax(net(test_img), dim=1)
    # test_class_str = [classes[i] for i in test_class]
    # label_str = [classes[i] for i in test_labels]
    # fig, ax = plt.subplots(6,6, figsize=(15,15),)
    # test_img = test_img *0.5 *255 + 0.5*255
    # test_img = test_img.to(torch.uint8)
    #
    # for i in range(6):
    #     for j in range(6):
    #         ax[i,j].imshow(test_img[i*6 + j].permute(1,2,0), interpolation="bicubic")
    #         ax[i,j].set_title(f"Pred: {test_class_str[i*6 + j]} True: {label_str[i*6 + j]}")
    #         ax[i,j].axis("off")
    # fig.savefig("./results.png")
    # plt.show()