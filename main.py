# Standard library imports
import logging
from typing import Annotated

# Third-party imports
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Local imports
from src.model.model import OneDSelfONN
from src.utils.log.manager import LoggerManager
from src.utils.data.manager import DataManager, DatasetLoader
from src.model.train import Trainer, InverseFrequencyClassWeighting


def main() -> Annotated[None, "This function does not return anything"]:
    """
    Load configuration from a YAML file, set up data, train specified
    model, and evaluate the final test performance.

    Returns
    -------
    None
        This function does not return anything.

    Examples
    --------
    >>> main()
    Initialize data managers, set up loaders, train the model, and
    evaluate its performance on the test set.
    """

    # Configuration
    config_path = "config/config.yaml"
    config = OmegaConf.load(config_path)
    device = config.device.type
    batch_size = config.train.onedselfonn.training.batch
    model = OneDSelfONN(config.train.onedselfonn.model).to(device)
    trainer_config = config.train.onedselfonn.training

    # Initialize classes
    logger = LoggerManager(console_level=logging.INFO).get_logger()
    data_manager = DataManager(
        signals_path=config.paths.signals,
        labels_path=config.paths.labels,
        csv_path=config.paths.csv,
        data_dir=config.paths.data,
        max_length=config.data.length,
        label_map=dict(config.data.labels),
        logger=logger
    )

    # Data split & load
    all_signals_np, all_labels_np = data_manager.load_data()
    logger.info(
        f"Main train data shape: {all_signals_np.shape}, {all_labels_np.shape}"
    )

    x_temp, x_test, y_temp, y_test = train_test_split(
        all_signals_np,
        all_labels_np,
        test_size=0.2,
        stratify=all_labels_np,
        random_state=config.data.splits.state
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=0.25,
        stratify=y_temp,
        random_state=config.data.splits.state
    )

    logger.info(f"Train shape: {x_train.shape}, {y_train.shape}")
    logger.info(f"Validation shape: {x_val.shape}, {y_val.shape}")
    logger.info(f"Test shape: {x_test.shape}, {y_test.shape}")

    train_dataset = DatasetLoader(x_train, y_train)
    val_dataset = DatasetLoader(x_val, y_val)
    test_dataset = DatasetLoader(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Inverse-Frequency Class Weighting & Loss
    unique_classes = np.unique(y_train)
    num_classes = len(unique_classes)
    class_counts = [np.sum(y_train == c) for c in unique_classes]
    logger.info(f"Class distributions (train): {class_counts}")

    ifcw = InverseFrequencyClassWeighting(
        y_train=y_train,
        num_classes=num_classes
    )
    class_weights_tensor = ifcw.get_weight_tensor(device=device)
    logger.info(f"Class weights (tensor) => {class_weights_tensor}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    logger.info("CrossEntropyLoss created (with Inverse-Frequency Weighting)")

    # Optimizer
    optimizer_cls = getattr(optim, trainer_config.optimizer)
    optimizer = optimizer_cls(model.parameters(), lr=trainer_config.lr)
    logger.info(f"Optimizer: {trainer_config.optimizer}, lr={trainer_config.lr}")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=trainer_config.epochs,
        targets=list(config.data.targets),
        val_loader=val_loader,
        test_loader=test_loader,
        logger=logger,
        plot_metrics=True,
        save_plot=True,
        plot_confusion_matrix=True,
        save_confusion_matrix=True,
        early_stopping=False,
        plot_roc=True,
        save_roc=True,
    )

    logger.info("Trainer created, training is starting...")
    torch.autograd.set_detect_anomaly(True)
    trainer.run()

    logger.info("=== Final Test Evaluation ===")
    trainer.evaluate(-1, test_loader, mode="Test")
    logger.info("=== Completed. ===")


if __name__ == "__main__":
    main()
