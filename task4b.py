
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
import torchvision
import torch
import numpy as np
image = Image.open("images/zebra.jpg")
print("Image shape before transformation:", image.size)
model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape after transformation:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image

def plot_filters(weights_tensor: torch.Tensor, filter_activation: torch.Tensor, indices: list, name: str):
    num_cols = len(indices)
    num_rows = 2
    plot_path = pathlib.Path("plots")
    fig = plt.figure(figsize=(num_cols, num_rows))
    for i in range(num_cols * num_rows):
        if i < num_cols:
            filter = torch_image_to_numpy(weights_tensor[indices[i]])
        else:
            filter = torch_image_to_numpy(filter_activation[0][indices[i - 5]])
        ax1 = fig.add_subplot(num_rows, num_cols, (i + 1) % ((num_cols * num_rows) + 1))
        ax1.imshow(filter)
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))


weights_tensor = first_conv_layer.weight.data
indices = [14, 26, 32, 49, 52]
plot_filters(weights_tensor, activation, indices, "task4b")
