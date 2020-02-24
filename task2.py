import pathlib
import matplotlib.pyplot as plt
import torch
import utils
import time
import typing
import torchvision
import collections
from torch import nn
from tqdm import tqdm
from dataloaders import load_cifar10


def compute_loss_and_accuracy(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_criterion: torch.nn.modules.loss._Loss):
    """
    Computes the average loss and the accuracy over the whole dataset
    in dataloader.
    Args:
        dataloder: Validation/Test dataloader
        model: torch.nn.Module
        loss_criterion: The loss criterion, e.g: torch.nn.CrossEntropyLoss()
    Returns:
        [average_loss, accuracy]: both scalar.
    """
    average_loss = 0
    accuracy = 0
    counter = 0
    correct_predictions = 0
    total_size = 0

    with torch.no_grad():
        for (X_batch, Y_batch) in dataloader:
            # Transfer images/labels to GPU VRAM, if possible
            X_batch = utils.to_cuda(X_batch)
            Y_batch = utils.to_cuda(Y_batch)
            # Forward pass the images through our model
            output_probs = model(X_batch)

            # Compute Loss
            average_loss += loss_criterion(output_probs, Y_batch)

            # Compute accuracy
            prediction = output_probs.argmax(dim=1)
            correct_predictions += (prediction == Y_batch).sum().item() 
            total_size += X_batch.shape[0]             
            counter+=1 
    accuracy = correct_predictions/total_size
    average_loss /= counter
    return average_loss, accuracy


class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.num_filters_cl1 = 32  # Set number of filters in first conv layer
        self.num_filters_cl2 = 64 # "" second conv layer 
        self.num_filters_cl3 = 128 # "" third conv layer 
        self.num_filters_fcl1 = 64 # number of filter in the first fully connected layer 
        self.num_classes = num_classes # second and last fully connected layer
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=self.num_filters_cl1,
                kernel_size=5,
                stride=1,
                padding=2
                ), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels= self.num_filters_cl1,
                    out_channels=self.num_filters_cl2,
                    kernel_size=5,
                    stride=1,
                    padding=2
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=self.num_filters_cl2, 
                    out_channels=self.num_filters_cl3, 
                    kernel_size=5, 
                    stride=1, 
                    padding=2
                ),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
       
        # The output of feature_extractor will be [batch_size, num_filters, 4, 4]
        self.num_output_features = 4*4*self.num_filters_cl3
        # Initialize our last fully connected layer
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        #self.classifier = nn.Sequential(
        #    nn.Linear(self.num_output_features, num_classes),
        #)
        # Define the fully connected layers (FCL)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, self.num_filters_fcl1),
            nn.ReLU(),
            nn.Linear(self.num_filters_fcl1, self.num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        x = self.feature_extractor(x)
        x = x.view(-1, self.num_output_features)
        x = self.classifier(x)
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


# Start experimenting for task 3, try to find the best parameters 
class Model7764(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.num_filters_cl1 = 32  # Set number of filters in first conv layer
        self.num_filters_cl2 = 64 # "" second conv layer 
        self.num_filters_cl3 = 128 # "" third conv layer
        self.num_filters_cl4 = 256  
        self.num_filters_fcl1 = 64 # number of filter in the first fully connected layer 
        self.num_classes = num_classes # second and last fully connected layer
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=self.num_filters_cl1,
                kernel_size=5,
                stride=1,
                padding=2
                ), 
            nn.ReLU(), 
            nn.Conv2d(in_channels= self.num_filters_cl1,
                    out_channels=self.num_filters_cl2,
                    kernel_size=5,
                    stride=1,
                    padding=2
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=self.num_filters_cl2, 
                    out_channels=self.num_filters_cl3, 
                    kernel_size=5, 
                    stride=1, 
                    padding=2
                ),
            nn.ReLU(), 
            nn.Conv2d(in_channels=self.num_filters_cl3, 
                    out_channels=self.num_filters_cl4, 
                    kernel_size=5, 
                    stride=1, 
                    padding=2
                ),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=self.num_filters_cl4, 
                    out_channels=self.num_filters_cl4, 
                    kernel_size=5, 
                    stride=1, 
                    padding=2
                ),
            nn.ReLU(), 
            nn.Conv2d(in_channels=self.num_filters_cl4, 
                    out_channels=self.num_filters_cl4, 
                    kernel_size=5, 
                    stride=1, 
                    padding=2
                ),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
       
        # The output of feature_extractor will be [batch_size, num_filters, 4, 4]
        # Find out size output after cvn (square): size_new = (size_old - F_old+2P_old)/S_old + 1
        self.num_output_features = 16*self.num_filters_cl4
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        # Define the fully connected layers (FCL)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, self.num_filters_fcl1),
            nn.ReLU(),
            nn.Linear(self.num_filters_fcl1, self.num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        x = self.feature_extractor(x)
        x = x.view(-1, self.num_output_features)
        x = self.classifier(x)
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

class Model7604(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.num_filters_cl1 = 64  # Set number of filters in first conv layer
        self.num_filters_cl2 = 128 # "" second conv layer 
        self.num_filters_cl3 = 256 # "" third conv layer 
        self.num_filters_fcl1 = 95 # number of filter in the first fully connected layer 
        self.num_classes = num_classes # second and last fully connected layer
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=self.num_filters_cl1,
                kernel_size=5,
                stride=1,
                padding=2
                ), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels= self.num_filters_cl1,
                    out_channels=self.num_filters_cl2,
                    kernel_size=5,
                    stride=1,
                    padding=2
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=self.num_filters_cl2, 
                    out_channels=self.num_filters_cl3, 
                    kernel_size=5, 
                    stride=1, 
                    padding=2
                ),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
       
        # The output of feature_extractor will be [batch_size, num_filters, 4, 4]
        # Find out size output after cvn (square): size_new = (size_old - F_old+2P_old)/S_old + 1
        self.num_output_features = 16*self.num_filters_cl3
        # Initialize our last fully connected layer
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        #self.classifier = nn.Sequential(
        #    nn.Linear(self.num_output_features, num_classes),
        #)
        # Define the fully connected layers (FCL)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, self.num_filters_fcl1),
            nn.ReLU(),
            nn.Linear(self.num_filters_fcl1, self.num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        x = self.feature_extractor(x)
        x = x.view(-1, self.num_output_features)
        x = self.classifier(x)
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

class Model7524(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.num_filters_cl1 = 64  # Set number of filters in first conv layer
        self.num_filters_cl2 = 128 # "" second conv layer 
        self.num_filters_cl3 = 256 # "" third conv layer
        self.num_filters_fcl1 = 95 # number of filter in the first fully connected layer 
        self.num_classes = num_classes # second and last fully connected layer
        # Define the convolutional layers
        # Try to use Adadelta as optimizer 
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=self.num_filters_cl1,
                kernel_size=5,
                stride=1,
                padding=2
                ), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels= self.num_filters_cl1,
                    out_channels=self.num_filters_cl2,
                    kernel_size=5,
                    stride=1,
                    padding=2
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=self.num_filters_cl2, 
                    out_channels=self.num_filters_cl3, 
                    kernel_size=5, 
                    stride=1, 
                    padding=2
                ),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
       
        # The output of feature_extractor will be [batch_size, num_filters, 4, 4]
        self.num_output_features = 4*4*self.num_filters_cl3
        # Initialize our last fully connected layer
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        #self.classifier = nn.Sequential(
        #    nn.Linear(self.num_output_features, num_classes),
        #)
        # Define the fully connected layers (FCL)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, self.num_filters_fcl1),
            nn.ReLU(),
            nn.Linear(self.num_filters_fcl1, self.num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        x = self.feature_extractor(x)
        x = x.view(-1, self.num_output_features)
        x = self.classifier(x)
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

# +
class Model7704(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.num_filters_cl1 = 64  # Set number of filters in first conv layer
        self.num_filters_cl2 = 128 # "" second conv layer 
        self.num_filters_cl3 = 256 # "" third conv layer
        self.num_filters_cl4 = 256
        self.num_filters_fcl1 = 95 # number of filter in the first fully connected layer 
        self.num_classes = num_classes # second and last fully connected layer
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=self.num_filters_cl1,
                kernel_size=5,
                stride=1,
                padding=2
                ), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels= self.num_filters_cl1,
                    out_channels=self.num_filters_cl2,
                    kernel_size=5,
                    stride=1,
                    padding=2
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=self.num_filters_cl2, 
                    out_channels=self.num_filters_cl3, 
                    kernel_size=5, 
                    stride=1, 
                    padding=2
                ),
            nn.ReLU(), 
            nn.Conv2d(in_channels=self.num_filters_cl3, 
                    out_channels=self.num_filters_cl4, 
                    kernel_size=5, 
                    stride=1, 
                    padding=2
                ),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
       
        # The output of feature_extractor will be [batch_size, num_filters, 4, 4]
        self.num_output_features = 4*4*self.num_filters_cl4
        # Initialize our last fully connected layer
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        #self.classifier = nn.Sequential(
        #    nn.Linear(self.num_output_features, num_classes),
        #)
        # Define the fully connected layers (FCL)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, self.num_filters_fcl1),
            nn.ReLU(),
            nn.Linear(self.num_filters_fcl1, self.num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        x = self.feature_extractor(x)
        x = x.view(-1, self.num_output_features)
        x = self.classifier(x)
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out
    
class Task3Model(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.num_filters_cl1 = 64  # Set number of filters in first conv layer
        self.num_filters_cl2 = 128 # "" second conv layer 
        self.num_filters_cl3 = 256 # "" third conv layer
        self.num_filters_cl4 = 256
        self.num_filters_fcl1 = 95 # number of filter in the first fully connected layer 
        self.num_classes = num_classes # second and last fully connected layer
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=self.num_filters_cl1,
                kernel_size=5,
                stride=1,
                padding=2
                ), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels= self.num_filters_cl1,
                    out_channels=self.num_filters_cl2,
                    kernel_size=5,
                    stride=1,
                    padding=2
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=self.num_filters_cl2, 
                    out_channels=self.num_filters_cl3, 
                    kernel_size=5, 
                    stride=1, 
                    padding=2
                ),
            nn.ReLU(), 
            nn.Conv2d(in_channels=self.num_filters_cl3, 
                    out_channels=self.num_filters_cl4, 
                    kernel_size=5, 
                    stride=1, 
                    padding=2
                ),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # The output of feature_extractor will be [batch_size, num_filters, 4, 4]
        self.num_output_features = 4*4*self.num_filters_cl4
        # Initialize our last fully connected layer
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        #self.classifier = nn.Sequential(
        #    nn.Linear(self.num_output_features, num_classes),
        #)
        # Define the fully connected layers (FCL)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, self.num_filters_fcl1),
            nn.ReLU(),
            nn.Linear(self.num_filters_fcl1, self.num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        x = self.feature_extractor(x)
        x = x.view(-1, self.num_output_features)
        x = self.classifier(x)
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


# -

class Trainer:

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 early_stop_count: int,
                 epochs: int,
                 model: torch.nn.Module,
                 dataloaders: typing.List[torch.utils.data.DataLoader]):
        """
            Initialize our trainer class.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_count = early_stop_count
        self.epochs = epochs

        # Since we are doing multi-class classification, we use CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the model
        self.model = model
        # Transfer model to GPU VRAM, if possible.
        self.model = utils.to_cuda(self.model)
        print(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.learning_rate)
        # Attempt a different optimizer, Adadelta should be used when using Model7524 
        #self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=0.5, rho=0.6, eps=1e-06, weight_decay=0)
        #self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
        
        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = dataloaders
        
        # Validate our model everytime we pass through 50% of the dataset
        self.num_steps_per_val = len(self.dataloader_train) // 2
        self.global_step = 0
        self.start_time = time.time()

        # Tracking variables
        self.VALIDATION_LOSS = collections.OrderedDict()
        self.TEST_LOSS = collections.OrderedDict()
        self.TRAIN_LOSS = collections.OrderedDict()
        self.VALIDATION_ACC = collections.OrderedDict()
        self.TEST_ACC = collections.OrderedDict()

        self.checkpoint_dir = pathlib.Path("checkpoints")

    def validation_epoch(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        self.VALIDATION_ACC[self.global_step] = validation_acc
        self.VALIDATION_LOSS[self.global_step] = validation_loss
        used_time = time.time() - self.start_time
        print(
            f"Epoch: {self.epoch:>2}",
            f"Batches per seconds: {self.global_step / used_time:.2f}",
            f"Global step: {self.global_step:>6}",
            f"Validation Loss: {validation_loss:.2f},",
            f"Validation Accuracy: {validation_acc:.3f}",
            sep="\t")
        # Compute for testing set
        test_loss, test_acc = compute_loss_and_accuracy(
            self.dataloader_test, self.model, self.loss_criterion
        )
        self.TEST_ACC[self.global_step] = test_acc
        self.TEST_LOSS[self.global_step] = test_loss

        self.model.train()

    def should_early_stop(self):
        """
            Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = list(self.VALIDATION_LOSS.values())[-self.early_stop_count:]
        first_loss = relevant_loss[0]
        if first_loss == min(relevant_loss):
            print("Early stop criteria met")
            return True
        return False

    def train(self, diff_optimizer=False):
        """
        Trains the model for [self.epochs] epochs.
        """
        # Track initial loss/accuracy
        def should_validate_model():
            return self.global_step % self.num_steps_per_val == 0

        for epoch in range(self.epochs):
            self.epoch = epoch
            # Perform a full pass through all the training samples
            for X_batch, Y_batch in self.dataloader_train:
                # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
                # Y_batch is the CIFAR10 image label. Shape: [batch_size]
                # Transfer images / labels to GPU VRAM, if possible
                X_batch = utils.to_cuda(X_batch)
                Y_batch = utils.to_cuda(Y_batch)

                # Perform the forward pass
                predictions = self.model(X_batch)
                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)
                self.TRAIN_LOSS[self.global_step] = loss.detach().cpu().item()

                # Backpropagation
                loss.backward()

                # Gradient descent step
                self.optimizer.step()

                # Reset all computed gradients to 0
                self.optimizer.zero_grad()
                self.global_step += 1
                 # Compute loss/accuracy for all three datasets.
                if should_validate_model():
                    self.validation_epoch()
                    self.save_model()
                    if self.should_early_stop():
                        print("Early stopping.")
                        self.print_outputs()
                        return
        self.print_outputs()

    def save_model(self):
        def is_best_model():
            """
                Returns True if current model has the lowest validation loss
            """
            validation_losses = list(self.VALIDATION_LOSS.values())
            return validation_losses[-1] == min(validation_losses)

        state_dict = self.model.state_dict()
        filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")

        utils.save_checkpoint(state_dict, filepath, is_best_model())

    def load_best_model(self):
        state_dict = utils.load_best_checkpoint(self.checkpoint_dir)
        if state_dict is None:
            print(
                f"Could not load best checkpoint. Did not find under: {self.checkpoint_dir}")
            return
        self.model.load_state_dict(state_dict)
    
    def print_outputs(self): 
        # Display the final results 
        train_average_loss, train_accuracy = compute_loss_and_accuracy(self.dataloader_train, self.model, self.loss_criterion)
        val_average_loss, val_accuracy = compute_loss_and_accuracy(self.dataloader_val, self.model, self.loss_criterion)
        test_average_loss, test_accuracy = compute_loss_and_accuracy(self.dataloader_test, self.model, self.loss_criterion)
        print("The final average train loss : {}".format(train_average_loss))
        print("The final train accuracy : {}".format(train_accuracy))
        print("The final average validations loss : {}".format(val_average_loss))
        print("The final validation accuracy : {}".format(val_accuracy))
        print("The final average test loss : {}".format(test_average_loss))
        print("The final test accuracy : {}".format(test_accuracy))


def create_plots(trainer1: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(2, 1, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer1.TRAIN_LOSS, label="Training loss")
    utils.plot_loss(trainer1.VALIDATION_LOSS, label="Validation loss")
    utils.plot_loss(trainer1.TEST_LOSS, label="Testing Loss")
    #utils.plot_loss(trainer2.TRAIN_LOSS, label="Training loss Best")
    #utils.plot_loss(trainer2.VALIDATION_LOSS, label="Validation loss Best")
    #utils.plot_loss(trainer2.TEST_LOSS, label="Testing Loss Best")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer1.VALIDATION_ACC, label="Validation Accuracy")
    utils.plot_loss(trainer1.TEST_ACC, label="Testing Accuracy")
    #utils.plot_loss(trainer2.VALIDATION_ACC, label="Validation Accuracy Best")
    #utils.plot_loss(trainer2.TEST_ACC, label="Testing Accuracy Best")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()
    return  






if __name__ == "__main__":
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = Model7704(image_channels=3, num_classes=10)
    trainer1 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer1.train()
    #trainer2 = Trainer(
    #    batch_size,
    #    learning_rate,
    #    early_stop_count,
    #    epochs,
    #    best_model,
    #    dataloaders
    #)
    #trainer2.train()

    create_plots(trainer1, "task3a")

