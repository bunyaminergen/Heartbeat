# Standard library imports
import os
import logging
from typing import Annotated, Dict, Optional, Tuple

# Third-party imports
import wfdb
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class DataManager:
    """
    DataManager handles loading of signals and labels from `.npy` files.
    If these files do not exist, it triggers the DataPreprocessor to
    generate them.

    This class is useful for ensuring data availability and consistency
    before any downstream processing or model training.

    Parameters
    ----------
    signals_path : str
        Path to the `.npy` file containing preprocessed signals.
    labels_path : str
        Path to the `.npy` file containing labels.
    csv_path : str
        Path to the reference CSV used during preprocessing.
    data_dir : str
        Directory where raw data (WFDB records) are stored.
    max_length : int
        Maximum length of each signal.
    label_map : dict
        Dictionary mapping label strings to integer classes.
    logger : logging.Logger
        Logger object used to log information, warnings, and errors.

    Attributes
    ----------
    signals_path : str
        Path to the signals file.
    labels_path : str
        Path to the labels file.
    csv_path : str
        CSV reference file path.
    data_dir : str
        Directory containing raw data.
    max_length : int
        Maximum length of signals.
    label_map : dict
        Mapping of labels to integer classes.
    logger : logging.Logger
        Logger for information output.

    Examples
    --------
    >>> import logging
    >>> logger = logging.getLogger(__name__)
    >>> dm = DataManager(
    ...     signals_path="path/to/signals.npy",
    ...     labels_path="path/to/labels.npy",
    ...     csv_path="path/to/reference.csv",
    ...     data_dir="path/to/raw/data",
    ...     max_length=9000,
    ...     label_map={"N": 0, "A": 1, "O": 2, "~": 3},
    ...     logger=logger
    ... )
    >>> signals, labels = dm.load_data()
    >>> signals.shape
    (100, 9000)
    >>> labels.shape
    (100,)
    """

    def __init__(
            self,
            signals_path: Annotated[str, "Path to the signals npy file"],
            labels_path: Annotated[str, "Path to the labels npy file"],
            csv_path: Annotated[str, "Path to the reference CSV file"],
            data_dir: Annotated[str, "Directory of raw data files"],
            max_length: Annotated[int, "Maximum length for each signal"],
            label_map: Annotated[Dict[str, int], "Mapping from label strings to ints"],
            logger: Annotated[logging.Logger, "Logger instance for logging"]
    ) -> None:
        """
        Initialize the DataManager with paths and preprocessing configuration.

        Parameters
        ----------
        signals_path : str
            Path to the `.npy` file containing preprocessed signals.
        labels_path : str
            Path to the `.npy` file containing labels.
        csv_path : str
            Path to the reference CSV file for preprocessing.
        data_dir : str
            Directory where raw WFDB data is stored.
        max_length : int
            Maximum length of each signal.
        label_map : dict
            Dictionary mapping label strings to integers.
        logger : logging.Logger
            Logger instance for logging messages.

        Raises
        ------
        TypeError
            If any argument is not of the expected type.
        """
        if not isinstance(signals_path, str):
            raise TypeError("Expected str for signals_path.")
        if not isinstance(labels_path, str):
            raise TypeError("Expected str for labels_path.")
        if not isinstance(csv_path, str):
            raise TypeError("Expected str for csv_path.")
        if not isinstance(data_dir, str):
            raise TypeError("Expected str for data_dir.")
        if not isinstance(max_length, int):
            raise TypeError("Expected int for max_length.")
        if not isinstance(label_map, dict):
            raise TypeError("Expected dict for label_map.")
        if not isinstance(logger, logging.Logger):
            raise TypeError("Expected a logging.Logger instance for logger.")

        self.signals_path = signals_path
        self.labels_path = labels_path
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.max_length = max_length
        self.label_map = label_map
        self.logger = logger

    def load_data(
            self
    ) -> Annotated[Tuple[np.ndarray, np.ndarray],
    "Tuple containing signals and labels arrays"]:
        """
        Load signals and labels from `.npy` files. If these files do not
        exist, run `DataPreprocessor` to generate them.

        Returns
        -------
        signals_np : np.ndarray
            Loaded signals array of shape (num_samples, max_length).
        labels_np : np.ndarray
            Corresponding labels array of shape (num_samples,).

        Raises
        ------
        FileNotFoundError
            If `signals_path` or `labels_path` is missing and cannot be
            created.

        Examples
        --------
        >>> signals, labels = DataManager.load_data()
        >>> signals.shape
        (100, 9000)
        >>> labels.shape
        (100,)
        """
        if not (os.path.exists(self.signals_path)
                and os.path.exists(self.labels_path)):
            self.logger.info(
                f"{self.signals_path} or {self.labels_path} "
                "not found. DataPreprocessor is running..."
            )
            preproc = DataPreprocessor(
                csv_path=self.csv_path,
                data_dir=self.data_dir,
                max_length=self.max_length,
                label_map=self.label_map,
                out_signals=self.signals_path,
                out_labels=self.labels_path
            )
            preproc.run()
        else:
            self.logger.info(
                f"{self.signals_path} and {self.labels_path} exist. "
                "They will be loaded directly..."
            )

        signals_np = np.load(self.signals_path)
        labels_np = np.load(self.labels_path)
        self.logger.info(
            f"Data size: signals={signals_np.shape}, "
            f"labels={labels_np.shape}"
        )

        return signals_np, labels_np


class DatasetLoader(Dataset):
    """
    DatasetLoader is a PyTorch Dataset that wraps signals and labels
    into a convenient iterable for data loading and batching.

    It provides a way to index samples and retrieve both the signal
    tensor and the corresponding label tensor.

    Parameters
    ----------
    signals : np.ndarray
        Numpy array of signal data.
    labels : np.ndarray
        Numpy array of label data (same length as `signals`).

    Attributes
    ----------
    signals : np.ndarray
        Stored signals data.
    labels : np.ndarray
        Stored labels data.
    length : int
        Number of samples in the dataset.

    Examples
    --------
    >>> signals = np.random.rand(10, 9000)
    >>> labels = np.random.randint(0, 4, size=10)
    >>> ds = DatasetLoader(signals, labels)
    >>> len(ds)
    10
    >>> x, y = ds[0]
    >>> x.shape
    torch.Size([1, 9000])
    >>> y.shape
    torch.Size([])
    """

    def __init__(
            self,
            signals: Annotated[np.ndarray, "Numpy array of signals"],
            labels: Annotated[np.ndarray, "Numpy array of labels"]
    ) -> None:
        """
        Initialize the DatasetLoader with signal and label data.

        Parameters
        ----------
        signals : np.ndarray
            Numpy array of signals.
        labels : np.ndarray
            Numpy array of labels. Must have the same length as signals.

        Raises
        ------
        TypeError
            If `signals` or `labels` is not a np.ndarray.
        ValueError
            If `signals` and `labels` have different lengths.
        """
        if not isinstance(signals, np.ndarray):
            raise TypeError("signals must be a np.ndarray.")
        if not isinstance(labels, np.ndarray):
            raise TypeError("labels must be a np.ndarray.")
        if signals.shape[0] != labels.shape[0]:
            raise ValueError(
                "signals and labels must have the same number of samples."
            )

        self.signals = signals
        self.labels = labels
        self.length = len(signals)

    def __len__(self) -> Annotated[int, "Number of samples in the dataset"]:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples.
        """
        return self.length

    def __getitem__(
            self,
            idx: Annotated[int, "Index of the sample to retrieve"]
    ) -> Annotated[Tuple[torch.Tensor, torch.Tensor],
    "Tuple containing signal tensor and label tensor"]:
        """
        Retrieve a single sample and its label by index.

        Parameters
        ----------
        idx : int
            Index to retrieve the data from.

        Returns
        -------
        x : torch.Tensor
            The signal at the given index, expanded to 2D shape
            (1, signal_length).
        y : torch.Tensor
            The label at the given index.

        Raises
        ------
        TypeError
            If `idx` is not an integer.

        """
        if not isinstance(idx, int):
            raise TypeError("Index (idx) must be an integer.")

        x = self.signals[idx]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class DataPreprocessor:
    """
    DataPreprocessor reads a reference CSV containing record paths and
    labels, loads raw WFDB signals, applies padding or truncation to
    match `max_length`, and saves the resulting signals and labels as
    `.npy` files.

    This class provides a reproducible method for preparing data prior
    to training or analysis.

    Parameters
    ----------
    csv_path : str
        Path to the reference CSV file containing record paths and labels.
    data_dir : str
        Directory containing the WFDB records.
    max_length : int
        Maximum length of each signal.
    label_map : dict, optional
        Mapping from label strings to integers. Defaults to
        {"N": 0, "A": 1, "O": 2, "~": 3}.
    out_signals : str, optional
        Path to the output `.npy` file for signals. Defaults to
        `"signals.npy"`.
    out_labels : str, optional
        Path to the output `.npy` file for labels. Defaults to
        `"labels.npy"`.

    Attributes
    ----------
    csv_path : str
        Reference CSV file path.
    data_dir : str
        Directory of raw WFDB data.
    max_length : int
        Maximum length of each signal.
    label_map : dict
        Mapping of label strings to integers.
    out_signals : str
        Output path for signals.
    out_labels : str
        Output path for labels.

    Examples
    --------
    >>> data_preprocessor = DataPreprocessor(
    ...     csv_path="path/to/REFERENCE.csv",
    ...     data_dir="path/to/raw_data",
    ...     max_length=9000
    ... )
    >>> preprocessor.run()
    >>> # Now you should see `signals.npy` and `labels.npy` created.
    """

    def __init__(
            self,
            csv_path: Annotated[str, "Path to the reference CSV"],
            data_dir: Annotated[str, "Directory of WFDB records"],
            max_length: Annotated[int, "Maximum length of each signal"],
            label_map: Annotated[
                Optional[Dict[str, int]],
                "Label mapping dictionary or None"
            ] = None,
            out_signals: Annotated[str, "Output path for signals `.npy`"] = "signals.npy",
            out_labels: Annotated[str, "Output path for labels `.npy`"] = "labels.npy"
    ) -> None:
        """
        Initialize the DataPreprocessor with paths and configuration.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing records and labels.
        data_dir : str
            Directory where the WFDB records are located.
        max_length : int
            Maximum length of each signal.
        label_map : dict, optional
            Mapping from label strings to integers.
        out_signals : str, optional
            Path for saving signals as a `.npy` file.
        out_labels : str, optional
            Path for saving labels as a `.npy` file.

        Raises
        ------
        TypeError
            If arguments are of incorrect types.
        """
        if not isinstance(csv_path, str):
            raise TypeError("csv_path must be a string.")
        if not isinstance(data_dir, str):
            raise TypeError("data_dir must be a string.")
        if not isinstance(max_length, int):
            raise TypeError("max_length must be an integer.")
        if label_map is not None and not isinstance(label_map, dict):
            raise TypeError("label_map must be a dictionary or None.")
        if not isinstance(out_signals, str):
            raise TypeError("out_signals must be a string.")
        if not isinstance(out_labels, str):
            raise TypeError("out_labels must be a string.")

        if label_map is None:
            label_map = {"N": 0, "A": 1, "O": 2, "~": 3}

        self.csv_path = csv_path
        self.data_dir = data_dir
        self.max_length = max_length
        self.label_map = label_map
        self.out_signals = out_signals
        self.out_labels = out_labels

    def run(self) -> None:
        """
        Execute the preprocessing steps:
          - Read the CSV reference file.
          - Load WFDB signals.
          - Pad or truncate signals to `max_length`.
          - Convert labels via `label_map`.
          - Save signals and labels to `.npy` files.

        Raises
        ------
        FileNotFoundError
            If the CSV file is not found or a record path is invalid.

        Examples
        --------
        >>> preprocessor.run()
        >>> # Generated files: signals.npy and labels.npy
        """
        print(f"Loading reference CSV: {self.csv_path}")
        df_ref = pd.read_csv(self.csv_path, header=None, sep=',')
        df_ref.columns = ["record_path", "label"]

        signals_list = []
        labels_list = []

        for _, row in df_ref.iterrows():
            record_relative = row["record_path"]
            label_str = row["label"]

            full_path = os.path.join(self.data_dir, record_relative)
            data, _ = wfdb.rdsamp(full_path)
            signal_1d = data[:, 0]
            length_signal = len(signal_1d)

            if length_signal < self.max_length:
                padded = np.pad(
                    signal_1d,
                    (0, self.max_length - length_signal),
                    constant_values=0
                )
            else:
                padded = signal_1d[:self.max_length]

            signals_list.append(padded)
            label_int = self.label_map[label_str]
            labels_list.append(label_int)

        signals_np = np.vstack(signals_list)
        labels_np = np.array(labels_list)

        print(f"Data shapes: {signals_np.shape}, {labels_np.shape}")
        print(f"Saving to: {self.out_signals}, {self.out_labels}")

        out_dir_signals = os.path.dirname(self.out_signals)
        out_dir_labels = os.path.dirname(self.out_labels)

        if out_dir_signals:
            os.makedirs(out_dir_signals, exist_ok=True)
        if out_dir_labels:
            os.makedirs(out_dir_labels, exist_ok=True)

        np.save(self.out_signals, signals_np)
        np.save(self.out_labels, labels_np)

        print("Completed. Files created successfully.")


if __name__ == "__main__":
    preprocessor = DataPreprocessor(
        csv_path=".data/raw/train/REFERENCE.csv",
        data_dir=".data/raw/train/",
        max_length=9000,
        label_map={"N": 0, "A": 1, "O": 2, "~": 3},
        out_signals=".data/binary/signals.npy",
        out_labels=".data/binary/labels.npy"
    )
    preprocessor.run()
