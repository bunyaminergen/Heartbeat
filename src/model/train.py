# Standard library imports
import copy
import logging
from typing import Annotated

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Local imports
from src.utils.visualization.visualize import Visualizer
from src.model.evaluation import Evaluation


class Trainer:
    """
    Trainer class that handles training, validation, and optional testing of a
    given PyTorch model.

    This class manages the training loop, validation, optional test evaluation,
    early stopping, and metric visualization.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to be trained.
    train_loader : DataLoader
        Dataloader providing the training dataset.
    criterion : nn.Module
        The loss function used for training.
    optimizer : torch.optim.Optimizer
        The optimizer for updating model parameters.
    device : str or torch.device
        The device on which computations will be performed.
    num_epochs : int, optional
        Number of epochs to train. Default is 19.
    targets : list of str, optional
        Class labels for evaluations. Default is None, which becomes
        ["Class_0", "Class_1", "Class_2", "Class_3"].
    val_loader : DataLoader, optional
        Dataloader for validation data. Default is None.
    logger : logging.Logger, optional
        Logger instance for logging messages. If None, prints to console.
    test_loader : DataLoader, optional
        Dataloader for test data. Default is None.
    plot_metrics : bool, optional
        If True, plot training and validation metrics. Default is False.
    save_plot : bool, optional
        If True, save metric plots to a file. Default is False.
    plot_file_name : str, optional
        File name to save the plots. Default is ".docs/report/img/training.png".
    plot_confusion_matrix : bool, optional
        If True, plot a confusion matrix. Default is False.
    save_confusion_matrix : bool, optional
        If True, save the confusion matrix plot. Default is False.
    early_stopping : bool, optional
        If True, enable early stopping. Default is True.
    patience : int, optional
        Number of epochs to wait for improvement when early stopping is
        enabled. Default is 5.
    print_epoch_reports : bool, optional
        If True, print evaluation reports after each epoch. Default is True.
    print_final_reports : bool, optional
        If True, print final evaluation reports. Default is True.
    plot_roc : bool, optional
        If True, plot ROC curves. Default is False.
    save_roc : bool, optional
        If True, save the ROC curves to a file. Default is False.
    roc_file_name : str, optional
        The file path to save the ROC curves. Default is ".docs/report/img/roc_curve.png".

    Attributes
    ----------
    model : nn.Module
        The PyTorch model.
    train_loader : DataLoader
        Training DataLoader.
    val_loader : DataLoader
        Validation DataLoader, if provided.
    test_loader : DataLoader
        Test DataLoader, if provided.
    optimizer : torch.optim.Optimizer
        Optimizer for model training.
    criterion : nn.Module
        Loss function for training.
    device : str or torch.device
        Training device.
    num_epochs : int
        Number of epochs to train.
    best_val_loss : float
        Best validation loss observed for early stopping.
    train_losses : list of float
        Track training loss per epoch.
    train_accs : list of float
        Track training accuracy per epoch.
    val_losses : list of float
        Track validation loss per epoch.
    val_accs : list of float
        Track validation accuracy per epoch.
    test_losses : list of float
        Track test loss if test_loader is used.
    test_accs : list of float
        Track test accuracy if test_loader is used.
    evaluation : Evaluation
        Evaluation object to handle metrics computation.

    Methods
    -------
    _log(msg, level="info")
        Log or print a message at the specified level.
    train(epoch_idx)
        Train the model for a single epoch.
    evaluate(epoch_idx, loader, mode="Val", print_reports=None)
        Evaluate the model using the specified dataloader.
    run()
        Run the full training loop with optional validation, early stopping,
        and test evaluation.
    _final_metrics()
        Compute and log final metrics for train, validation, and test sets.

    Examples
    --------
    >>> model = nn.Linear(10, 2)
    >>> X_dummy_test = torch.randn(100, 10)
    >>> y_dummy_test = torch.randint(0, 2, (100,))
    >>> train_ds_test = TensorDataset(X_dummy_test, y_dummy_test)
    >>> train_loader = DataLoader(train_ds_test, batch_size=8)
    >>> criterion = nn.CrossEntropyLoss()
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> device = "cpu"
    >>> trainer_test = Trainer(model, train_loader_test,
    ...                        criterion, optimizer,
    ...                        device, num_epochs=2)
    >>> trainer_test.run()
    """

    def __init__(
            self,
            model: Annotated[nn.Module, "PyTorch model to be trained"],
            train_loader: Annotated[DataLoader, "Training DataLoader"],
            criterion: Annotated[nn.Module, "Loss function"],
            optimizer: Annotated[torch.optim.Optimizer, "Optimizer for training"],
            device: Annotated[str, "Device (e.g., 'cpu' or 'cuda')"],
            num_epochs: Annotated[int, "Number of training epochs"] = 19,
            targets: Annotated[list[str], "Class labels"] = None,
            val_loader: Annotated[DataLoader, "Validation DataLoader"] = None,
            logger: Annotated[logging.Logger, "Logger instance"] = None,
            test_loader: Annotated[DataLoader, "Test DataLoader"] = None,
            plot_metrics: Annotated[bool, "Whether to plot training metrics"] = False,
            save_plot: Annotated[bool, "Whether to save plot to file"] = False,
            plot_file_name: Annotated[str, "Plot file name"] = ".docs/report/img/training.png",
            plot_confusion_matrix: Annotated[bool, "Whether to plot a confusion matrix"] = False,
            save_confusion_matrix: Annotated[bool, "Whether to save the confusion matrix"] = False,
            early_stopping: Annotated[bool, "Enable early stopping"] = True,
            patience: Annotated[int, "Patience for early stopping"] = 5,
            print_epoch_reports: Annotated[bool, "Print reports at each epoch"] = True,
            print_final_reports: Annotated[bool, "Print final reports"] = True,
            plot_roc: Annotated[bool, "Whether to plot ROC curves"] = False,
            save_roc: Annotated[bool, "Whether to save ROC plots"] = False,
            roc_file_name: Annotated[str, "File path to save the ROC curves"] = ".docs/report/img/roc_curve.png"
    ) -> None:
        """
        Initialize the Trainer class with the given configuration parameters.

        (Docstring içeriği korunmuştur, sadece yeni parametreler eklenmiştir.)
        """
        if not isinstance(model, nn.Module):
            raise TypeError("Expected 'model' to be an instance of nn.Module")
        if not isinstance(train_loader, DataLoader):
            raise TypeError("Expected 'train_loader' to be a DataLoader")
        if not isinstance(criterion, nn.Module):
            raise TypeError("Expected 'criterion' to be an instance of nn.Module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("Expected 'optimizer' to be a torch.optim.Optimizer")
        if not (isinstance(device, str) or isinstance(device, torch.device)):
            raise TypeError("Expected 'device' to be str or torch.device")
        if not isinstance(num_epochs, int):
            raise TypeError("Expected 'num_epochs' to be an int")

        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.logger = logger
        self.plot_metrics = plot_metrics
        self.save_plot = save_plot
        self.plot_file_name = plot_file_name
        self.plot_confusion_matrix = plot_confusion_matrix
        self.save_confusion_matrix = save_confusion_matrix
        self.early_stopping = early_stopping
        self.patience = patience
        self.print_epoch_reports = print_epoch_reports
        self.print_final_reports = print_final_reports

        self.plot_roc = plot_roc
        self.save_roc = save_roc
        self.roc_file_name = roc_file_name

        if targets is None:
            targets = [f"Class_{i}" for i in range(4)]
        self.targets = targets

        self.visualizer = Visualizer(logger=self.logger)

        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.best_model_weights = None

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.test_losses = []
        self.test_accs = []

        self.evaluation = Evaluation(targets=self.targets, logger=self.logger)

    def _log(
            self,
            msg: Annotated[str, "Message to log"],
            level: Annotated[str, "Logging level"] = "info"
    ) -> None:
        """
        Log or print the given message at the specified logging level.

        Parameters
        ----------
        msg : str
            Message to be logged.
        level : str, optional
            Logging level ("info", "debug", "warning", "error").
            Default is "info".

        Examples
        --------
        >>> self._log("This is an info message")
        """
        if not isinstance(msg, str):
            raise TypeError("Expected 'msg' to be a string")
        if not isinstance(level, str):
            raise TypeError("Expected 'level' to be a string")

        if self.logger is not None:
            if level == "info":
                self.logger.info(msg)
            elif level == "debug":
                self.logger.debug(msg)
            elif level == "warning":
                self.logger.warning(msg)
            elif level == "error":
                self.logger.error(msg)
        else:
            print(msg)

    def train(
            self,
            epoch_idx: Annotated[int, "Current epoch index"]
    ) -> Annotated[tuple[float, float], "Tuple of (epoch_loss, epoch_accuracy)"]:
        """
        Train the model for a single epoch.

        This method iterates over the training DataLoader, performs forward and
        backward passes, computes the loss, and updates the model parameters.

        Parameters
        ----------
        epoch_idx : int
            The index of the current epoch (0-based).

        Returns
        -------
        (epoch_loss, epoch_accuracy) : tuple of float
            A tuple containing the average loss and accuracy for this epoch.

        Examples
        --------
        >>> loss_test, acc_test = self.train(0)
        >>> print(loss_test, acc_test)
        1.234 0.5
        """
        if not isinstance(epoch_idx, int):
            raise TypeError("Expected 'epoch_idx' to be an int")

        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        epoch_loss = float(running_loss / len(self.train_loader))
        epoch_acc = float(np.mean(all_preds == all_labels))

        self._log(
            f"[Train] Epoch {epoch_idx + 1} - Loss: {epoch_loss:.4f}, "
            f"Acc: {epoch_acc:.4f}"
        )
        return epoch_loss, epoch_acc

    def evaluate(
            self,
            epoch_idx: Annotated[int, "Current epoch index"],
            loader: Annotated[DataLoader, "DataLoader to evaluate"],
            mode: Annotated[str, "Evaluation mode (e.g., 'Val' or 'Test')"] = "Val",
            print_reports: Annotated[bool, "Whether to print reports"] = None
    ) -> Annotated[tuple[float, float], "Tuple of (loss, accuracy)"]:
        """
        Evaluate the model using the given DataLoader.

        This method delegates evaluation to the Evaluation class, computes
        loss and accuracy, and can optionally print detailed reports.

        Parameters
        ----------
        epoch_idx : int
            The index of the current epoch (0-based).
        loader : DataLoader
            The DataLoader providing data for evaluation.
        mode : str, optional
            The evaluation mode label ("Val", "Train", "Test"). Default is "Val".
        print_reports : bool, optional
            Whether to print detailed evaluation reports. If None, defaults
            to self.print_epoch_reports.

        Returns
        -------
        loss : float
            The average loss computed over the given DataLoader.
        accuracy : float
            The average accuracy computed over the given DataLoader.

        Examples
        --------
        >>> val_loss, val_acc = self.evaluate(0, loader=self.val_loader)
        >>> print(val_loss, val_acc)
        0.876 0.55
        """
        if not isinstance(epoch_idx, int):
            raise TypeError("Expected 'epoch_idx' to be an int")
        if not isinstance(loader, DataLoader):
            raise TypeError("Expected 'loader' to be a DataLoader")
        if not isinstance(mode, str):
            raise TypeError("Expected 'mode' to be a string")
        if print_reports is not None and not isinstance(print_reports, bool):
            raise TypeError("Expected 'print_reports' to be a bool or None")

        if print_reports is None:
            print_reports = self.print_epoch_reports

        loss, accuracy = self.evaluation.evaluate(
            model=self.model,
            loader=loader,
            criterion=self.criterion,
            device=self.device,
            epoch_idx=epoch_idx,
            mode=mode,
            reports=print_reports
        )

        return float(loss), float(accuracy)

    def run(self) -> None:
        """
        Run the full training loop with optional validation, early stopping,
        and test evaluation.

        This method iterates through the specified number of epochs, calling
        train and evaluate. If early stopping is enabled and no improvement in
        validation loss is observed for 'patience' epochs, training stops
        early. Optionally, final metrics and confusion matrices are plotted
        or saved.

        Examples
        --------
        >>> self.run()
        """
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            val_loss, val_acc = 0.0, 0.0
            if self.val_loader is not None:
                val_loss, val_acc = self.evaluate(epoch, self.val_loader)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)

            test_loss, test_acc = 0.0, 0.0
            if self.test_loader is not None:
                test_loss, test_acc = self.evaluate(
                    epoch, self.test_loader, mode="Test", print_reports=False
                )
                self.test_losses.append(test_loss)
                self.test_accs.append(test_acc)

            if self.early_stopping and self.val_loader is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0
                    self.best_model_weights = copy.deepcopy(self.model.state_dict())
                else:
                    self.early_stop_counter += 1
                    self._log(
                        f"No improvement. EarlyStopCounter: "
                        f"{self.early_stop_counter}/{self.patience}"
                    )
                    if self.early_stop_counter >= self.patience:
                        self._log(
                            f"Early stopping triggered at epoch {epoch + 1}. "
                            f"Best val_loss: {self.best_val_loss:.4f}"
                        )
                        break

        self._log("Training completed.")

        if self.best_model_weights is not None:
            self._log("Loading best model weights from early stopping checkpoint.")
            self.model.load_state_dict(self.best_model_weights)

        if self.plot_metrics or self.save_plot:
            self.visualizer.plot_loss_accuracy(
                train_losses=self.train_losses,
                train_accs=self.train_accs,
                val_losses=self.val_losses if self.val_loader else None,
                val_accs=self.val_accs if self.val_loader else None,
                test_losses=self.test_losses if self.test_loader else None,
                test_accs=self.test_accs if self.test_loader else None,
                save_plot=self.save_plot,
                plot_file_name=self.plot_file_name,
                show_plot=self.plot_metrics
            )

        if self.print_final_reports:
            self._final_metrics()

    def _final_metrics(self) -> None:
        """
        Compute and log final metrics for training, validation, and test sets.

        This method evaluates the model on train, validation, and test
        dataloaders (if provided), prints the results, and optionally plots
        confusion matrices.

        Examples
        --------
        >>> self._final_metrics()
        """
        self._log("=== Final Aggregated Results ===")

        if self.train_loader is not None:
            self._log("---- Final Train Evaluation ----")
            final_train_loss, final_train_acc = self.evaluate(
                epoch_idx=self.num_epochs,
                loader=self.train_loader,
                mode="Train",
                print_reports=True
            )
            self._log(
                f"Final Train Loss: {final_train_loss:.4f}, "
                f"Final Train Accuracy: {final_train_acc:.4f}"
            )

            if self.plot_confusion_matrix:
                self.visualizer.plot_confusion_matrix_final(
                    model=self.model,
                    loader=self.train_loader,
                    device=self.device,
                    targets=self.targets,
                    title="Final Train Confusion Matrix",
                    file_name=".docs/report/img/confusion_matrix_train.png",
                    save_cm=self.save_confusion_matrix,
                    show_cm=self.plot_metrics
                )

        if self.val_loader is not None:
            self._log("---- Final Validation Evaluation ----")
            final_val_loss, final_val_acc = self.evaluate(
                epoch_idx=self.num_epochs,
                loader=self.val_loader,
                print_reports=True
            )
            self._log(
                f"Final Validation Loss: {final_val_loss:.4f}, "
                f"Final Validation Accuracy: {final_val_acc:.4f}"
            )

            if self.plot_confusion_matrix:
                self.visualizer.plot_confusion_matrix_final(
                    model=self.model,
                    loader=self.val_loader,
                    device=self.device,
                    targets=self.targets,
                    title="Final Validation Confusion Matrix",
                    file_name=".docs/report/img/confusion_matrix_val.png",
                    save_cm=self.save_confusion_matrix,
                    show_cm=self.plot_metrics
                )

        if self.test_loader is not None:
            self._log("---- Final Test Evaluation ----")
            final_test_loss, final_test_acc = self.evaluate(
                epoch_idx=self.num_epochs,
                loader=self.test_loader,
                mode="Test",
                print_reports=True
            )
            self.test_losses.append(final_test_loss)
            self.test_accs.append(final_test_acc)

            self._log(
                f"Final Test Loss: {final_test_loss:.4f}, "
                f"Final Test Accuracy: {final_test_acc:.4f}"
            )

            if self.plot_confusion_matrix:
                self.visualizer.plot_confusion_matrix_final(
                    model=self.model,
                    loader=self.test_loader,
                    device=self.device,
                    targets=self.targets,
                    title="Final Test Confusion Matrix",
                    file_name=".docs/report/img/confusion_matrix_test.png",
                    save_cm=self.save_confusion_matrix,
                    show_cm=self.plot_metrics
                )

            if self.plot_roc:
                self.visualizer.plot_roc_curves_final(
                    model=self.model,
                    loader=self.test_loader,
                    device=self.device,
                    targets=self.targets,
                    file_name=self.roc_file_name,
                    save_roc=self.save_roc,
                    show_roc=self.plot_roc,
                    logger=self.logger
                )

        self._log("=== Final Detailed Evaluation Completed. ===")


class InverseFrequencyClassWeighting:
    """
    Compute class weights using the inverse frequency method.

    This class calculates class weights based on their inverse frequencies
    in the training data. These weights are typically used for imbalanced
    classification tasks.

    Parameters
    ----------
    y_train : numpy.ndarray
        The training labels as a 1D array.
    num_classes : int
        The total number of classes.

    Attributes
    ----------
    y_train : numpy.ndarray
        The training labels.
    num_classes : int
        Number of classes.

    Methods
    -------
    compute_weights()
        Compute the inverse frequency weights for each class.
    get_weight_tensor(device='cpu')
        Return the weights as a torch.Tensor on the specified device.

    Examples
    --------
    >>> y_data = np.array([0, 0, 1, 2, 2, 2])
    >>> weighting = InverseFrequencyClassWeighting(y_data, 3)
    >>> class_weights = weighting.compute_weights()
    >>> print(class_weights)
    [1.0, 1.5, 0.75]
    >>> weight_tensor = weighting.get_weight_tensor(device='cuda')
    >>> print(weight_tensor)
    tensor([1.0000, 1.5000, 0.7500])
    """

    def __init__(
            self,
            y_train: Annotated[np.ndarray, "Array of training labels"],
            num_classes: Annotated[int, "Number of classes"]
    ) -> None:
        """
        Initialize the InverseFrequencyClassWeighting instance.

        Parameters
        ----------
        y_train : numpy.ndarray
            The training labels as a 1D array.
        num_classes : int
            Number of classes to consider for weighting.
        """
        if not isinstance(y_train, np.ndarray):
            raise TypeError("Expected 'y_train' to be a numpy.ndarray")
        if not isinstance(num_classes, int):
            raise TypeError("Expected 'num_classes' to be an int")

        self.y_train = y_train
        self.num_classes = num_classes

    def compute_weights(self) -> Annotated[list[float], "List of computed weights"]:
        """
        Compute the class weights based on inverse frequency.

        Returns
        -------
        weights : list of float
            The computed inverse frequency weights for each class.

        Examples
        --------
        >>> y_data = np.array([0, 0, 1, 2, 2, 2])
        >>> weighting = InverseFrequencyClassWeighting(y_data, 3)
        >>> class_weights = weighting.compute_weights()
        >>> class_weights
        [1.0, 1.5, 0.75]
        """
        unique_classes = np.unique(self.y_train)
        class_counts = []
        for c in unique_classes:
            class_counts.append(np.sum(self.y_train == c))

        total = len(self.y_train)
        weights = []
        for count in class_counts:
            weights.append(total / (count * self.num_classes))

        return weights

    def get_weight_tensor(
            self,
            device: Annotated[str, "Device to place the weight tensor"] = 'cpu'
    ) -> Annotated[torch.Tensor, "Weight tensor for PyTorch"]:
        """
        Convert the computed weights into a PyTorch tensor.

        Parameters
        ----------
        device : str, optional
            The device where the weight tensor will be placed. Default is 'cpu'.

        Returns
        -------
        weight_tensor : torch.Tensor
            A tensor containing the inverse frequency weights.

        Examples
        --------
        >>> y_data = np.array([0, 0, 1, 2, 2, 2])
        >>> weighting = InverseFrequencyClassWeighting(y_data, 3)
        >>> weight_tensors = weighting.get_weight_tensor(device='cuda')
        >>> weight_tensors
        tensor([1.0000, 1.5000, 0.7500])
        """
        if not isinstance(device, str) and not isinstance(device, torch.device):
            raise TypeError("Expected 'device' to be a str or torch.device")

        weights = self.compute_weights()
        weight_tensor = torch.tensor(weights, dtype=torch.float).to(device)
        return weight_tensor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger_test = logging.getLogger("TrainerExample")


    class SimpleModel(nn.Module):
        """
        A simple model for demonstration purposes.

        This class defines a single convolutional layer followed by an
        adaptive average pooling and a linear layer. It outputs logits for
        four classes.

        Examples
        --------
        >>> import torch
        >>> model = SimpleModel()
        >>> dummy_input = torch.randn(1, 1, 128)
        >>> output = model(dummy_input)
        >>> print(output.shape)
        torch.Size([1, 4])
        """

        def __init__(self) -> None:
            """
            Initialize the SimpleModel.

            This model consists of:
            - A 1D convolution layer with 1 input channel and 4 output channels,
              kernel size 3.
            - An adaptive average pooling layer to reduce feature dimensions.
            - A linear layer to produce four class logits.
            """
            super().__init__()
            self.conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(4, 4)

        def forward(
                self,
                x: Annotated[torch.Tensor, "Input tensor of shape (N, 1, L)"]
        ) -> Annotated[torch.Tensor, "Output logits of shape (N, 4)"]:
            """
            Forward pass of the SimpleModel.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape (N, 1, L).

            Returns
            -------
            torch.Tensor
                Output logits of shape (N, 4).

            Examples
            --------
            >>> model = SimpleModel()
            >>> dummy_input = torch.randn(1, 1, 128)
            >>> output = model(dummy_input)
            >>> output.shape
            torch.Size([1, 4])
            """
            if not isinstance(x, torch.Tensor):
                raise TypeError("Expected input 'x' to be a torch.Tensor")

            x = self.conv(x)
            x = self.pool(x)
            x = x.squeeze(-1)
            x = self.fc(x)
            return x


    X_dummy = torch.randn(100, 1, 128)
    y_dummy = torch.randint(0, 4, (100,))

    train_ds = TensorDataset(X_dummy[:80], y_dummy[:80])
    val_ds = TensorDataset(X_dummy[80:], y_dummy[80:])

    train_loader_test = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader_test = DataLoader(val_ds, batch_size=8, shuffle=False)

    model_test = SimpleModel()
    criterion_test = nn.CrossEntropyLoss()
    optimizer_test = torch.optim.Adam(model_test.parameters(), lr=0.001)
    device_test = "cpu"

    trainer = Trainer(
        model=model_test,
        train_loader=train_loader_test,
        criterion=criterion_test,
        optimizer=optimizer_test,
        device=device_test,
        num_epochs=5,
        targets=["Class0", "Class1", "Class2", "Class3"],
        val_loader=val_loader_test,
        logger=logger_test,
        plot_metrics=True,
        plot_confusion_matrix=True,
        early_stopping=False,
        plot_roc=True,
        save_roc=True,
    )

    trainer.run()
    logger_test.info("Trainer run is complete.")
