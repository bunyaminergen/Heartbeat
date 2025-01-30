# Standard library imports
import logging
from typing import Annotated, Optional, Callable

# Third-party imports
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


class Evaluation:
    """
    The Evaluation class provides methods for evaluating a PyTorch model's
    performance on a given dataset. It computes the loss and accuracy for
    each epoch and can also generate classification reports and confusion
    matrices.

    This class is useful for tasks where you need to measure model performance
    over multiple epochs and optionally log detailed metrics such as OVR AUC,
    classification reports, and confusion matrices.

    Parameters
    ----------
    targets : list of str, optional
        List of target class names. If not provided, defaults to
        ['Class_0', 'Class_1', 'Class_2', 'Class_3'].
    logger : logging.Logger, optional
        A logger instance for recording messages. If None, messages will be
        printed to standard output.

    Attributes
    ----------
    targets : list of str
        The list of target class names.
    logger : logging.Logger or None
        The logger instance for recording messages or None if no logger is
        used.

    Methods
    -------
    _log(msg, level='info')
        Logs or prints a message using the provided logger instance.
    evaluate(model, loader, criterion, device, epoch_idx=0, mode='Val',
             reports=True)
        Evaluate the model on the dataset, calculating loss and accuracy,
        and optionally generating classification reports and confusion
        matrices.

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> from torch.utils.data import DataLoader, TensorDataset
    >>> dataset = TensorDataset(torch.randn(8, 4), torch.randint(0, 2, (8,)))
    >>> loader_ = DataLoader(dataset, batch_size=2)
    >>> model_ = nn.Sequential(nn.Linear(4, 2))
    >>> criterion_ = nn.CrossEntropyLoss()
    >>> evaluator = Evaluation()
    >>> epoch_loss_, epoch_acc_ = evaluator.evaluate(
    ...     model_,
    ...     loader_,
    ...     criterion_,
    ...     device='auto'
    ... )
    >>> print(epoch_loss_, epoch_acc_)
    """

    def __init__(
            self,
            targets: Annotated[
                Optional[list[str]],
                "List of class names, or None to use default"
            ] = None,
            logger: Annotated[
                Optional[logging.Logger],
                "Logger instance or None if no logger is used"
            ] = None
    ) -> None:
        """
        Initialize the Evaluation class with optional targets and logger.

        Parameters
        ----------
        targets : list of str, optional
            List of target class names. If not provided, defaults to
            ['Class_0', 'Class_1', 'Class_2', 'Class_3'].
        logger : logging.Logger, optional
            A logger instance for recording messages. If None, messages are
            printed to standard output.

        Examples
        --------
        >>> evaluator = Evaluation()
        >>> print(evaluator.targets)
        ['Class_0', 'Class_1', 'Class_2', 'Class_3']
        """
        if targets is None:
            targets = [f"Class_{i}" for i in range(4)]
        self.targets = targets
        self.logger = logger

    def _log(
            self,
            msg: Annotated[str, "Message to log or print"],
            level: Annotated[str, "Log level (info, debug, warning, error)"] = "info"
    ) -> None:
        """
        Log or print a message at the specified level.

        Parameters
        ----------
        msg : str
            The message to be logged or printed.
        level : str, optional
            The log level at which to record the message. Can be 'info',
            'debug', 'warning', or 'error'. Defaults to 'info'.

        Returns
        -------
        None

        Examples
        --------
        >>> evaluator = Evaluation()
        >>> evaluator._log("Hello, world!")
        Hello, world!
        """
        if not isinstance(msg, str):
            raise TypeError("Expected str for parameter 'msg'")
        if not isinstance(level, str):
            raise TypeError("Expected str for parameter 'level'")
        if level not in ["info", "debug", "warning", "error"]:
            raise ValueError(
                "Log level must be one of 'info', 'debug', 'warning', or 'error'."
            )

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

    def evaluate(
            self,
            model: Annotated[torch.nn.Module, "PyTorch model to evaluate"],
            loader: Annotated[
                torch.utils.data.DataLoader,
                "Data loader with input data and labels"
            ],
            criterion: Annotated[Callable, "Loss function for computing the loss"],
            device: Annotated[str, "One of 'cpu', 'cuda', or 'auto'"],
            epoch_idx: Annotated[int, "Current epoch index"] = 0,
            mode: Annotated[str, "Evaluation mode (e.g., 'Val')"] = "Val",
            reports: Annotated[bool, "If True, generate classification reports"] = True
    ) -> Annotated[tuple[float, float], "Tuple of (epoch_loss, epoch_accuracy)"]:
        """
        Evaluate the model on the provided dataset.

        This method computes the average loss and accuracy for one pass
        of the dataset. When requested, it also logs a classification
        report and a confusion matrix. Additionally, OVR (One-vs-Rest) AUC
        values are calculated for each class and reported.

        Parameters
        ----------
        model : torch.nn.Module
            PyTorch model to evaluate.
        loader : torch.utils.data.DataLoader
            Data loader providing input data and labels.
        criterion : Callable
            Loss function for computing the loss.
        device : str
            One of 'cpu', 'cuda', or 'auto'. If 'auto', will choose 'cuda'
            if available, otherwise 'cpu'.
        epoch_idx : int, optional
            Current epoch index. Defaults to 0.
        mode : str, optional
            Evaluation mode (e.g., 'Val', 'Train', 'Test'). Defaults to 'Val'.
        reports : bool, optional
            If True, generate a classification report and confusion matrix.
            Defaults to True.

        Returns
        -------
        tuple of float
            A tuple containing the epoch loss and accuracy, in that order.

        Examples
        --------
        >>> import torch
        >>> from torch import nn
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> dataset = TensorDataset(torch.randn(8, 4), torch.randint(0, 4, (8,)))
        >>> loader_ = DataLoader(dataset, batch_size=2)
        >>> model_ = nn.Sequential(nn.Linear(4, 4))
        >>> criterion_ = nn.CrossEntropyLoss()
        >>> evaluator = Evaluation(targets=["C0", "C1", "C2", "C3"])
        >>> epoch_loss_, epoch_acc_ = evaluator.evaluate(
        ...     model_,
        ...     loader_,
        ...     criterion_,
        ...     device='auto'
        ... )
        >>> print(epoch_loss_, epoch_acc_)
        """
        if device not in ["cpu", "cuda", "auto"]:
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'")

        if device == "auto":
            device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device_obj = torch.device(device)

        model = model.to(device_obj)

        if not hasattr(loader, "__iter__"):
            raise TypeError("Expected an iterable DataLoader for parameter 'loader'")
        if not callable(criterion):
            raise TypeError(
                "Expected a callable loss function for parameter 'criterion'"
            )

        if not isinstance(epoch_idx, int):
            raise TypeError("Expected int for parameter 'epoch_idx'")
        if not isinstance(mode, str):
            raise TypeError("Expected str for parameter 'mode'")
        if not isinstance(reports, bool):
            raise TypeError("Expected bool for parameter 'reports'")

        model.eval()
        running_loss = 0.0
        preds_list = []
        labels_list = []
        probs_list = []

        with torch.no_grad():
            for inp, lbl in loader:
                inp = inp.to(device_obj)
                lbl = lbl.to(device_obj)

                outputs = model(inp)
                loss = criterion(outputs, lbl)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                preds_list.append(predicted.cpu().numpy())
                labels_list.append(lbl.cpu().numpy())

                softmaxed = torch.softmax(outputs, dim=1).cpu().numpy()
                probs_list.append(softmaxed)

        preds = np.concatenate(preds_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        probs = np.concatenate(probs_list, axis=0)

        epoch_loss = float(running_loss / len(loader))
        epoch_acc = float(np.mean(preds == labels))

        self._log(
            f"[{mode}] Epoch {epoch_idx + 1} - Loss: {epoch_loss:.4f}, "
            f"Acc: {epoch_acc:.4f}"
        )

        num_classes = probs.shape[1]
        for class_idx in range(num_classes):
            labels_bin = np.where(labels == class_idx, 1, 0)
            probs_bin = probs[:, class_idx]
            try:
                auc_val = roc_auc_score(labels_bin, probs_bin)
                class_name = (
                    self.targets[class_idx]
                    if class_idx < len(self.targets)
                    else f"Class_{class_idx}"
                )
                self._log(f"[{mode}] AUC[One-vs-Rest (OvR)] for '{class_name}': {auc_val:.4f}")
            except ValueError as e:
                self._log(
                    f"[{mode}] AUC[One-vs-Rest (OvR)] could not be calculated "
                    f"(class_idx={class_idx}): {e}"
                )

        try:
            macro_auc = roc_auc_score(labels, probs, multi_class="ovr")
            self._log(f"[{mode}] AUC[One-vs-Rest (OvR)] (macro): {macro_auc:.4f}")
        except ValueError as e:
            self._log(f"[{mode}] AUC[One-vs-Rest (OvR)] could not be calculated: {e}")

        if reports:
            report = classification_report(
                labels,
                preds,
                target_names=self.targets,
                digits=4,
                zero_division=0
            )
            conf_matrix = confusion_matrix(labels, preds)

            self._log(f"Classification Report:\n{report}")
            self._log(f"Confusion Matrix:\n{conf_matrix}")

        return epoch_loss, epoch_acc


if __name__ == "__main__":
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    data_test = torch.randn(16, 4)
    labels_test = torch.randint(0, 4, (16,))
    dataset_test = TensorDataset(data_test, labels_test)
    loader_test = DataLoader(dataset_test, batch_size=4)

    model_test = nn.Sequential(nn.Linear(4, 4))
    criterion_test = nn.CrossEntropyLoss()

    evaluator_test = Evaluation(
        targets=["Class_0", "Class_1", "Class_2", "Class_3"]
    )
    epoch_loss_test, epoch_acc_test = evaluator_test.evaluate(
        model=model_test,
        loader=loader_test,
        criterion=criterion_test,
        device="auto"
    )

    print(f"Final Loss: {epoch_loss_test:.4f}")
    print(f"Final Accuracy: {epoch_acc_test:.4f}")
