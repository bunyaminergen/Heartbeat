# Standard library imports
import os
import logging
from typing import Annotated, Optional, List, Union

# Third party imports
import torch
import numpy as np
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset


class Visualizer:
    """
    Visualizer class for plotting training metrics and confusion matrices.

    This class stores an optional logger for informational messages and provides
    methods to visualize the training process (loss and accuracy curves) and
    to generate confusion matrices for classification tasks.

    Parameters
    ----------
    logger : logging.Logger, optional
        A logger instance for logging messages. If None is provided, no logging
        is performed.

    Attributes
    ----------
    logger : logging.Logger or None
        Logger instance for logging messages.
    """

    def __init__(
            self,
            logger: Annotated[Optional[logging.Logger], "Logger for logging messages"] = None
    ) -> None:
        """
        Initialize the Visualizer with an optional logger.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for informational messages. Defaults to None.
        """
        self.logger = logger

    def plot_loss_accuracy(
            self,
            train_losses: Annotated[
                Union[List[float], np.ndarray], "Training losses per epoch"
            ],
            train_accs: Annotated[
                Union[List[float], np.ndarray], "Training accuracies per epoch"
            ],
            val_losses: Annotated[
                Optional[Union[List[float], np.ndarray]], "Validation losses"
            ] = None,
            val_accs: Annotated[
                Optional[Union[List[float], np.ndarray]], "Validation accuracies"
            ] = None,
            test_losses: Annotated[
                Optional[Union[List[float], np.ndarray]], "Test losses"
            ] = None,
            test_accs: Annotated[
                Optional[Union[List[float], np.ndarray]], "Test accuracies"
            ] = None,
            save_plot: Annotated[bool, "Whether to save the plot"] = False,
            plot_file_name: Annotated[
                str, "File name for saving the plot"
            ] = ".docs/img/training_curves.png",
            show_plot: Annotated[bool, "Whether to display the plot"] = False,
            logger: Annotated[
                Optional[logging.Logger], "Logger for logging within this method"
            ] = None
    ) -> None:
        """
        Plot and optionally save/display loss and accuracy curves.

        This method plots training loss and accuracy, and optionally validation
        and test loss/accuracy, across epochs.

        Parameters
        ----------
        train_losses : list of float or np.ndarray
            Training losses per epoch.
        train_accs : list of float or np.ndarray
            Training accuracies per epoch.
        val_losses : list of float or np.ndarray, optional
            Validation losses per epoch. Defaults to None.
        val_accs : list of float or np.ndarray, optional
            Validation accuracies per epoch. Defaults to None.
        test_losses : list of float or np.ndarray, optional
            Test losses per epoch. Defaults to None.
        test_accs : list of float or np.ndarray, optional
            Test accuracies per epoch. Defaults to None.
        save_plot : bool, optional
            If True, saves the plot to `plot_file_name`. Defaults to False.
        plot_file_name : str, optional
            The file path to save the plot. Defaults to
            ".docs/img/training_curves.png".
        show_plot : bool, optional
            If True, displays the plot. Defaults to False.
        logger : logging.Logger, optional
            Logger instance for logging messages. If None, uses the class-level
            logger.

        Returns
        -------
        None
            This function does not return anything.

        Examples
        --------
        >>> visualizer = Visualizer()
        >>> train_losses_example = [0.6, 0.5, 0.4]
        >>> train_accs_example = [0.7, 0.75, 0.8]
        >>> visualizer.plot_loss_accuracy(train_losses_example,
        ...                               train_accs_example)
        """
        if not isinstance(train_losses, (list, np.ndarray)):
            raise TypeError("train_losses must be a list or NumPy array.")

        if not isinstance(train_accs, (list, np.ndarray)):
            raise TypeError("train_accs must be a list or NumPy array.")

        if val_losses is not None and not isinstance(val_losses, (list, np.ndarray)):
            raise TypeError("val_losses must be a list or NumPy array if provided.")

        if val_accs is not None and not isinstance(val_accs, (list, np.ndarray)):
            raise TypeError("val_accs must be a list or NumPy array if provided.")

        if test_losses is not None and not isinstance(test_losses, (list, np.ndarray)):
            raise TypeError("test_losses must be a list or NumPy array if provided.")

        if test_accs is not None and not isinstance(test_accs, (list, np.ndarray)):
            raise TypeError("test_accs must be a list or NumPy array if provided.")

        logger = logger or self.logger

        epochs_range = range(1, len(train_losses) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
        if val_losses is not None and len(val_losses) == len(train_losses):
            plt.plot(epochs_range, val_losses, label='Val Loss', marker='x')
        if test_losses is not None and len(test_losses) == len(train_losses):
            plt.plot(epochs_range, test_losses, label='Test Loss', marker='s')
        plt.title('Loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_accs, label='Train Acc', marker='o')
        if val_accs is not None and len(val_accs) == len(train_accs):
            plt.plot(epochs_range, val_accs, label='Val Acc', marker='x')
        if test_accs is not None and len(test_accs) == len(train_accs):
            plt.plot(epochs_range, test_accs, label='Test Acc', marker='s')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()

        if save_plot:
            dir_name = os.path.dirname(plot_file_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            plt.savefig(plot_file_name, dpi=150)
            if logger:
                logger.info(
                    f"[Visualizer] Saved training curves to: {plot_file_name}"
                )

        if show_plot:
            plt.show()

        plt.close()

    def plot_confusion_matrix_final(
            self,
            model: Annotated[nn.Module, "PyTorch model"],
            loader: Annotated[
                torch.utils.data.DataLoader, "DataLoader providing input and labels"
            ],
            device: Annotated[str, "Device for computation (e.g., 'cpu' or 'cuda')"],
            targets: Annotated[List[str], "List of class labels for ticks"],
            title: Annotated[str, "Title of the confusion matrix plot"] = "Confusion Matrix",
            file_name: Annotated[
                str, "File name for saving the confusion matrix figure"
            ] = ".docs/img/confusion_matrix.png",
            save_cm: Annotated[bool, "Whether to save the confusion matrix plot"] = False,
            show_cm: Annotated[bool, "Whether to display the confusion matrix plot"] = False,
            logger: Annotated[
                Optional[logging.Logger], "Logger for logging within this method"
            ] = None
    ) -> None:
        """
        Generate and optionally save/display a confusion matrix.

        This method evaluates the given model on the provided DataLoader,
        computes the confusion matrix, and plots it as a heatmap.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to evaluate.
        loader : torch.utils.data.DataLoader
            DataLoader providing (input, label) batches.
        device : str
            The device on which to perform computation (e.g., 'cpu', 'cuda').
        targets : list of str
            Class labels for x-axis and y-axis ticks.
        title : str, optional
            Title of the confusion matrix plot. Defaults to "Confusion Matrix".
        file_name : str, optional
            File path to save the confusion matrix plot. Defaults to
            ".docs/img/confusion_matrix.png".
        save_cm : bool, optional
            If True, saves the confusion matrix plot. Defaults to False.
        show_cm : bool, optional
            If True, displays the confusion matrix plot. Defaults to False.
        logger : logging.Logger, optional
            Logger instance for logging messages. If None, uses the class-level
            logger.

        Returns
        -------
        None
            This function does not return anything.

        Examples
        --------
        >>> import torch
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> model_test = torch.nn.Linear(10, 2)
        >>> data = torch.randn(20, 10)
        >>> labels_test = torch.randint(0, 2, (20,))
        >>> dataset_test = TensorDataset(data, labels_test)
        >>> loader_test = DataLoader(dataset_test, batch_size=5)
        >>> visualizer = Visualizer()
        >>> visualizer.plot_confusion_matrix_final(
        ...     model_test,
        ...     loader_test,
        ...     device='cpu',
        ...     targets=['Class 0', 'Class 1']
        ... )
        """
        if not hasattr(model, 'eval'):
            raise TypeError("model must be a PyTorch module with an 'eval' method.")

        if not hasattr(loader, '__iter__'):
            raise TypeError("loader must be an iterable DataLoader.")

        if not isinstance(device, str):
            raise TypeError("device must be a string ('cpu' or 'cuda').")

        if not isinstance(targets, list):
            raise TypeError("targets must be a list of class labels.")

        logger = logger or self.logger

        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, dim=1)
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=targets,
            yticklabels=targets
        )
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        if save_cm:
            dir_name = os.path.dirname(file_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            plt.savefig(file_name, dpi=150)
            if logger:
                logger.info(f"[Visualizer] Saved confusion matrix to: {file_name}")

        if show_cm:
            plt.show()

        plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger_test = logging.getLogger(__name__)

    visualizer_test = Visualizer(logger=logger_test)

    epochs = 10
    train_losses_test = np.random.rand(epochs) * 0.1 + 0.5
    train_accss_test = np.random.rand(epochs) * 0.1 + 0.8
    val_lossess_test = np.random.rand(epochs) * 0.1 + 0.4
    val_accss_test = np.random.rand(epochs) * 0.1 + 0.85

    visualizer_test.plot_loss_accuracy(
        train_losses=train_losses_test,
        train_accs=train_accss_test,
        val_losses=val_lossess_test,
        val_accs=val_accss_test,
        save_plot=True,
        plot_file_name=".docs/img/training_curves_example.png",
        show_plot=True,
    )

    logger_test.info("Loss/Accuracy chart logging completed.")


    class SimpleModel(nn.Module):
        """
        SimpleModel demonstrates a minimal PyTorch model for testing purposes.

        This class contains a single linear layer and serves as an example
        for passing a model into the Visualizer's methods.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input features.
        num_classes : int
            Number of output classes.

        Attributes
        ----------
        linear : nn.Linear
            A linear layer that projects input features to the given number of
            classes.

        Methods
        -------
        forward(x)
            Forward pass of the model.
        """

        def __init__(
                self,
                input_dim: Annotated[int, "Dimensionality of input features"],
                num_classes: Annotated[int, "Number of output classes"]
        ):
            """
            Initialize SimpleModel with a single linear layer.

            Parameters
            ----------
            input_dim : int
                Dimensionality of the input features.
            num_classes : int
                Number of output classes.
            """
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(input_dim, num_classes)

        def forward(
                self,
                x: Annotated[torch.Tensor, "Input tensor"]
        ) -> Annotated[torch.Tensor, "Output logits from the linear layer"]:
            """
            Forward pass of the model.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape (batch_size, input_dim).

            Returns
            -------
            torch.Tensor
                Output logits of shape (batch_size, num_classes).
            """
            return self.linear(x)


    input_dims_test = 5
    num_classess_test = 3
    models_test = SimpleModel(input_dims_test, num_classess_test)

    x_data = torch.randn(50, input_dims_test)
    y_data = torch.randint(0, num_classess_test, (50,))

    dataset = TensorDataset(x_data, y_data)
    loaders_test = DataLoader(dataset, batch_size=10)

    device_test = "cpu"
    models_test.to(device_test)

    class_names = ["Class0", "Class1", "Class2"]
    visualizer_test.plot_confusion_matrix_final(
        model=models_test,
        loader=loaders_test,
        device=device_test,
        targets=class_names,
        title="Confusion Matrix Example",
        file_name=".docs/img/confusion_matrix_example.png",
        save_cm=True,
        show_cm=True,
    )

    logger_test.info("Confusion matrix creation and saving completed.")
