# Standard library imports
import logging
from typing import Optional, Annotated

# Third-party imports
import pywt
import numpy as np
from scipy.signal import butter, filtfilt, zpk2tf


class Normalize:
    """
    Class providing various normalization and signal-processing methods
    for 2D segment data of shape (N, window).

    This class offers methods such as peak normalization, z-score,
    robust scaling, rank-Gauss transformation, wavelet detrending,
    high-pass filtering, realignment (placeholder), time warping
    (placeholder), and L2 normalization. Each method can operate in
    a 'local' or 'global' mode.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger for informational output. If None, messages will be
        printed to stdout.

    Attributes
    ----------
    logger : logging.Logger or None
        The logger instance for debug/info messages.
    """

    def __init__(
            self,
            logger: Annotated[Optional[logging.Logger],
            "Logger for informational output"] = None
    ) -> None:
        """
        Initialize the Normalize class with an optional logger.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance for informational output. If None, messages
            will be printed to stdout.

        Raises
        ------
        TypeError
            If the provided logger is not a logging.Logger or None.
        """
        if logger is not None and not isinstance(logger, logging.Logger):
            raise TypeError("logger must be a logging.Logger or None.")
        self.logger = logger

    def _log(self, msg: str) -> None:
        """
        Log a message using the provided logger or print to stdout.

        Parameters
        ----------
        msg : str
            The message to log or print.
        """
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def peak(
            self,
            segments: Annotated[np.ndarray,
            "Input segments of shape (N, window)"],
            mode: Annotated[str, "Normalization mode: 'local' or 'global'"] = 'local',
            global_absmax: Annotated[Optional[float],
            "Global absolute max for 'global' mode"] = None
    ) -> Annotated[np.ndarray, "Peak-normalized segments"]:
        """
        Perform peak normalization on each segment.

        Parameters
        ----------
        segments : numpy.ndarray
            2D NumPy array of shape (N, window).
        mode : {'local', 'global'}, optional
            - 'local': Normalize each segment independently by its own max.
            - 'global': Normalize all segments by a shared absolute maximum.
        global_absmax : float, optional
            Absolute maximum value used for 'global' normalization.
            Must be > 1e-12 if used.

        Returns
        -------
        numpy.ndarray
            Peak-normalized segments of the same shape as the input.

        Raises
        ------
        TypeError
            If segments is not a NumPy array.
        ValueError
            If segments is not 2-dimensional or mode is invalid.
        ValueError
            If global_absmax is not provided or <= 1e-12 in 'global' mode.

        Examples
        --------
        >>> norm = Normalize()
        >>> data = np.array([[1, 2, 3], [2, 4, 6]], dtype=np.float32)
        >>> norm.peak(data)
        array([[0.33333334, 0.6666667 , 1.        ],
               [0.33333334, 0.6666667 , 1.        ]], dtype=float32)

        >>> absmax = np.max(np.abs(data))
        >>> norm.peak(data, mode='global', global_absmax=absmax)
        array([[0.33333334, 0.6666667 , 1.        ],
               [0.6666667 , 1.3333334 , 2.        ]], dtype=float32)
        """
        if not isinstance(segments, np.ndarray):
            raise TypeError("segments must be a NumPy array.")
        if segments.ndim != 2:
            raise ValueError("Expected shape (N, window).")

        out = segments.astype(np.float32).copy()
        n_seg = out.shape[0]

        if mode == 'local':
            for i in range(n_seg):
                row = out[i]
                m = np.max(np.abs(row))
                if m > 1e-12:
                    out[i] = row / m
            self._log(f"[peak local] Applied to {n_seg} segments.")

        elif mode == 'global':
            if global_absmax is None or global_absmax < 1e-12:
                raise ValueError(
                    "global_absmax must be provided and > 1e-12 for 'global' mode."
                )
            out /= global_absmax
            self._log(f"[peak global] shape={out.shape}, global_absmax={global_absmax}")

        else:
            raise ValueError("mode must be 'local' or 'global'.")

        return out

    def zscore(
            self,
            segments: Annotated[np.ndarray,
            "Input segments of shape (N, window)"],
            mode: Annotated[str, "Z-score mode: 'local' or 'global'"] = 'local',
            global_mean: Annotated[Optional[float],
            "Global mean for 'global' mode"] = None,
            global_std: Annotated[Optional[float],
            "Global standard deviation for 'global' mode"] = None
    ) -> Annotated[np.ndarray, "Z-score normalized segments"]:
        """
        Perform z-score normalization on each segment.

        Parameters
        ----------
        segments : numpy.ndarray
            2D NumPy array of shape (N, window).
        mode : {'local', 'global'}, optional
            - 'local': Compute mean and std for each segment independently.
            - 'global': Use provided global mean and std for all segments.
        global_mean : float, optional
            Global mean to use for 'global' mode.
        global_std : float, optional
            Global standard deviation to use for 'global' mode.

        Returns
        -------
        numpy.ndarray
            Z-score normalized segments of the same shape as the input.

        Raises
        ------
        TypeError
            If segments is not a NumPy array.
        ValueError
            If segments is not 2-dimensional or mode is invalid.
        ValueError
            If global_mean or global_std is not provided for 'global' mode,
            or global_std <= 1e-12.

        Examples
        --------
        >>> norm = Normalize()
        >>> data = np.array([[1, 2, 3], [2, 4, 6]], dtype=np.float32)
        >>> norm.zscore(data)
        array([[-1.2247449 ,  0.        ,  1.2247449 ],
               [-1.2247449 ,  0.        ,  1.2247449 ]], dtype=float32)

        >>> g_mean = float(np.mean(data))
        >>> g_std = float(np.std(data))
        >>> norm.zscore(data, mode='global', global_mean=g_mean, global_std=g_std)
        array([[-1.0690448, -0.2672616,  0.5345225],
               [-0.5345225,  0.5345225,  1.6035676]], dtype=float32)
        """
        if not isinstance(segments, np.ndarray):
            raise TypeError("segments must be a NumPy array.")
        if segments.ndim != 2:
            raise ValueError("Expected shape (N, window).")

        out = segments.astype(np.float32).copy()
        n_seg = out.shape[0]

        if mode == 'local':
            for i in range(n_seg):
                row = out[i]
                m = np.mean(row)
                s = np.std(row)
                if s < 1e-12:
                    s = 1e-12
                out[i] = (row - m) / s
            self._log(f"[zscore local] Applied to {n_seg} segments.")

        elif mode == 'global':
            if global_mean is None or global_std is None:
                raise ValueError("Need global_mean and global_std for global zscore.")
            if global_std < 1e-12:
                global_std = 1e-12
            out = (out - global_mean) / global_std
            self._log(
                f"[zscore global] shape={out.shape}, mean={global_mean}, std={global_std}"
            )

        else:
            raise ValueError("mode must be 'local' or 'global'.")

        return out

    def robust(
            self,
            segments: Annotated[np.ndarray,
            "Input segments of shape (N, window)"],
            mode: Annotated[str, "Robust mode: 'local' or 'global'"] = 'local',
            global_median: Annotated[Optional[float],
            "Global median for 'global' mode"] = None,
            global_iqr: Annotated[Optional[float],
            "Global IQR for 'global' mode"] = None
    ) -> Annotated[np.ndarray, "Robust-scaled segments"]:
        """
        Perform robust scaling (median and IQR) on each segment.

        Parameters
        ----------
        segments : numpy.ndarray
            2D NumPy array of shape (N, window).
        mode : {'local', 'global'}, optional
            - 'local': Compute median and IQR for each segment independently.
            - 'global': Use provided global median and IQR for all segments.
        global_median : float, optional
            Global median to use for 'global' mode.
        global_iqr : float, optional
            Global IQR to use for 'global' mode. Must be > 1e-12.

        Returns
        -------
        numpy.ndarray
            Robust-scaled segments of the same shape as the input.

        Raises
        ------
        TypeError
            If segments is not a NumPy array.
        ValueError
            If segments is not 2-dimensional or mode is invalid.
        ValueError
            If global_median or global_iqr is not provided for 'global' mode,
            or global_iqr <= 1e-12.

        Examples
        --------
        >>> norm = Normalize()
        >>> data = np.array([[1, 2, 3], [2, 4, 6]], dtype=np.float32)
        >>> norm.robust(data)
        array([[-1.,  0.,  1.],
               [-1.,  0.,  1.]], dtype=float32)
        """
        if not isinstance(segments, np.ndarray):
            raise TypeError("segments must be a np.ndarray.")
        if segments.ndim != 2:
            raise ValueError("Expected shape (N, window).")

        out = segments.astype(np.float32).copy()
        n_seg = out.shape[0]

        if mode == 'local':
            for i in range(n_seg):
                row = out[i]
                med = np.median(row)
                q1 = np.percentile(row, 25)
                q3 = np.percentile(row, 75)
                iqr_ = q3 - q1
                if iqr_ < 1e-12:
                    iqr_ = 1e-12
                out[i] = (row - med) / iqr_
            self._log(f"[robust local] Applied to {n_seg} segments.")

        elif mode == 'global':
            if global_median is None or global_iqr is None:
                raise ValueError("Need global_median and global_iqr for robust global.")
            if global_iqr < 1e-12:
                global_iqr = 1e-12
            out = (out - global_median) / global_iqr
            self._log(
                f"[robust global] shape={out.shape}, median={global_median}, iqr={global_iqr}"
            )

        else:
            raise ValueError("mode must be 'local' or 'global'.")

        return out

    def rankgauss(
            self,
            segments: Annotated[np.ndarray,
            "Input segments of shape (N, window)"],
            mode: Annotated[str, "RankGauss mode: 'local' or 'global'"] = 'local'
    ) -> Annotated[np.ndarray, "RankGauss-transformed segments"]:
        """
        Apply rank-based Gaussianization (RankGauss) to each segment.

        Parameters
        ----------
        segments : numpy.ndarray
            2D NumPy array of shape (N, window).
        mode : {'local', 'global'}, optional
            - 'local': Transform each segment independently.
            - 'global': Transform all data together.

        Returns
        -------
        numpy.ndarray
            RankGauss-transformed segments of the same shape as the input.

        Raises
        ------
        TypeError
            If segments is not a NumPy array.
        ValueError
            If segments is not 2-dimensional or mode is invalid.
        ImportError
            If scipy.special.erfinv is not installed.

        Examples
        --------
        >>> norm = Normalize()
        >>> data = np.array([[10, 20, 30], [2, 4, 6]], dtype=np.float32)
        >>> rg_local = norm.rankgauss(data)
        >>> rg_local.shape
        (2, 3)

        >>> rg_global = norm.rankgauss(data, mode='global')
        >>> rg_global.shape
        (2, 3)
        """
        if not isinstance(segments, np.ndarray):
            raise TypeError("segments must be a np.ndarray.")
        if segments.ndim != 2:
            raise ValueError("Expected shape (N, window).")

        try:
            from scipy.special import erfinv
        except ImportError:
            raise ImportError("scipy.special.erfinv is required for rankgauss")

        out = segments.astype(np.float32).copy()
        n_seg, length = out.shape

        if mode == 'local':
            for i in range(n_seg):
                row = out[i]
                order = np.argsort(row)
                ranks = np.zeros_like(row, dtype=np.int32)
                ranks[order] = np.arange(length)
                cdf = (ranks.astype(np.float32) + 1) / (length + 1)
                row_gauss = np.sqrt(2.0).astype(np.float32) * \
                            erfinv(2.0 * cdf - 1.0)
                out[i] = row_gauss

            self._log(f"[rankgauss local] Applied to {n_seg} segments.")

        elif mode == 'global':
            flat_data = out.reshape(-1)
            order = np.argsort(flat_data)
            ranks = np.zeros_like(flat_data, dtype=np.int32)
            ranks[order] = np.arange(flat_data.size)
            cdf = (ranks.astype(np.float32) + 1) / (flat_data.size + 1)
            transformed_flat = np.sqrt(2.0).astype(np.float32) * \
                               erfinv(2.0 * cdf - 1.0)
            out = transformed_flat.reshape(n_seg, length)
            self._log(f"[rankgauss global] Flatten size={flat_data.size}, "
                      f"shape={out.shape}")

        else:
            raise ValueError("mode must be 'local' or 'global'.")

        return out

    def waveletdetrend(
            self,
            segments: Annotated[np.ndarray,
            "Input segments of shape (N, window)"],
            wavelet_name: Annotated[str,
            "Wavelet name for PyWavelets"] = 'db4',
            mode: Annotated[str, "Mode: 'local' or 'global'"] = 'local'
    ) -> Annotated[np.ndarray, "Wavelet-detrended segments"]:
        """
        Detrend segments using a wavelet-based approach.

        Parameters
        ----------
        segments : numpy.ndarray
            2D NumPy array of shape (N, window).
        wavelet_name : str, optional
            Name of the wavelet to use (e.g., 'db4').
        mode : {'local', 'global'}, optional
            Currently, 'global' is implemented the same as 'local'.

        Returns
        -------
        numpy.ndarray
            Wavelet-detrended segments of the same shape as the input.

        Raises
        ------
        ImportError
            If pywt is not installed.
        TypeError
            If segments is not a NumPy array.
        ValueError
            If segments is not 2-dimensional or mode is invalid.

        Examples
        --------
        >>> norm = Normalize()
        >>> data = np.array([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=np.float32)
        >>> norm.waveletdetrend(data, wavelet_name='db1').shape
        (2, 4)
        """
        if pywt is None:
            raise ImportError("pywt is required for waveletdetrend.")
        if not isinstance(segments, np.ndarray):
            raise TypeError("segments must be a np.ndarray.")
        if segments.ndim != 2:
            raise ValueError("Expected shape (N, window).")

        out = segments.astype(np.float32).copy()
        n_seg, length = out.shape

        if mode not in ('local', 'global'):
            raise ValueError("mode must be 'local' or 'global'.")

        if mode == 'global':
            self._log("[waveletdetrend] 'global' mode: using same method as local")

        for i in range(n_seg):
            row = out[i]
            ca, cd = pywt.dwt(row, wavelet_name)
            ca[:] = 0.0
            rec = pywt.idwt(ca, cd, wavelet_name)
            if len(rec) >= length:
                rec = rec[:length]
            else:
                pad_len = length - len(rec)
                rec = np.pad(rec, (0, pad_len))
            out[i] = rec.astype(np.float32)

        self._log(f"[waveletdetrend] wavelet={wavelet_name}, mode={mode}, "
                  f"applied to {n_seg} segments.")
        return out

    def highpass(
            self,
            segments: Annotated[np.ndarray,
            "Input segments of shape (N, window)"],
            cutoff: Annotated[float, "Cutoff frequency"] = 0.5,
            fs: Annotated[float, "Sampling rate (Hz)"] = 300.0,
            min_len: Annotated[int, "Minimum length required for filtering"] = 30,
            mode: Annotated[str, "Mode: 'local' or 'global'"] = 'local'
    ) -> Annotated[np.ndarray, "High-pass filtered segments"]:
        """
        Apply a high-pass filter to each segment.

        Parameters
        ----------
        segments : numpy.ndarray
            2D NumPy array of shape (N, window).
        cutoff : float, optional
            Cutoff frequency for the high-pass filter (Hz).
        fs : float, optional
            Sampling rate in Hz.
        min_len : int, optional
            Minimum length required to apply the filter. If a segment
            is shorter than this length, it is left unchanged.
        mode : {'local', 'global'}, optional
            Currently, 'global' is the same as 'local'.

        Returns
        -------
        numpy.ndarray
            High-pass filtered segments of the same shape.

        Raises
        ------
        ImportError
            If required scipy.signal functions are not installed.
        TypeError
            If segments is not a NumPy array.
        ValueError
            If segments is not 2-dimensional or mode is invalid.

        Examples
        --------
        >>> norm = Normalize()
        >>> data = np.tile(np.array([1, 2, 3, 4], dtype=np.float32), (2,1))
        >>> filtered = norm.highpass(data, cutoff=1.0, fs=10.0)
        >>> filtered.shape
        (2, 4)
        """
        if butter is None or filtfilt is None or zpk2tf is None:
            raise ImportError(
                "scipy.signal.butter/filtfilt/zpk2tf required for highpass."
            )
        if not isinstance(segments, np.ndarray):
            raise TypeError("segments must be a np.ndarray.")
        if segments.ndim != 2:
            raise ValueError("Expected shape (N, window).")

        if mode not in ('local', 'global'):
            raise ValueError("mode must be 'local' or 'global'.")
        if mode == 'global':
            self._log("[highpass] 'global' mode does not differ; "
                      "using same logic as local.")

        out = segments.astype(np.float32).copy()
        nyq = 0.5 * fs
        normal_cut = cutoff / nyq

        z, p, _ = butter(4, normal_cut, btype='high', output='zpk')
        b, a = zpk2tf(z, p, 1.0)

        n_seg = out.shape[0]

        for i in range(n_seg):
            row = out[i]
            length_ = len(row)
            if length_ < min_len:
                row_f = row
            else:
                try:
                    row_f = filtfilt(b, a, row, method='gust')
                except TypeError:
                    row_f = filtfilt(b, a, row)
            out[i] = row_f.astype(np.float32)

        self._log(f"[highpass] cutoff={cutoff}Hz, fs={fs}, min_len={min_len}, "
                  f"mode={mode}, segments={n_seg}.")
        return out

    def ralign(
            self,
            segments: Annotated[np.ndarray,
            "Input segments of shape (N, window)"],
            mode: Annotated[str, "Mode: 'local' or 'global'"] = 'local'
    ) -> Annotated[np.ndarray, "Realigned segments (placeholder)"]:
        """
        Realign segments (placeholder method).

        Parameters
        ----------
        segments : numpy.ndarray
            2D NumPy array of shape (N, window).
        mode : {'local', 'global'}, optional
            Currently, 'global' is the same as 'local'.

        Returns
        -------
        numpy.ndarray
            Realigned segments (same as input for now).

        Raises
        ------
        TypeError
            If segments is not a NumPy array.
        ValueError
            If segments is not 2-dimensional or mode is invalid.

        Examples
        --------
        >>> norm = Normalize()
        >>> data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        >>> aligned = norm.ralign(data)
        >>> aligned.shape
        (2, 3)
        """
        if not isinstance(segments, np.ndarray):
            raise TypeError("segments must be a np.ndarray.")
        if segments.ndim != 2:
            raise ValueError("Expected shape (N, window).")

        if mode not in ('local', 'global'):
            raise ValueError("mode must be 'local' or 'global'.")
        if mode == 'global':
            self._log("[ralign] global mode is same as local for now...")

        out = segments.astype(np.float32).copy()
        n_seg = out.shape[0]
        self._log(f"[ralign] (placeholder) mode={mode}, for {n_seg} segments.")
        return out

    def timewarp(
            self,
            segments: Annotated[np.ndarray,
            "Input segments of shape (N, window)"],
            mode: Annotated[str, "Mode: 'local' or 'global'"] = 'local'
    ) -> Annotated[np.ndarray, "Time-warped segments (placeholder)"]:
        """
        Apply time warping to segments (placeholder method).

        Parameters
        ----------
        segments : numpy.ndarray
            2D NumPy array of shape (N, window).
        mode : {'local', 'global'}, optional
            Currently, 'global' is the same as 'local'.

        Returns
        -------
        numpy.ndarray
            Time-warped segments (same as input for now).

        Raises
        ------
        TypeError
            If segments is not a NumPy array.
        ValueError
            If segments is not 2-dimensional or mode is invalid.

        Examples
        --------
        >>> norm = Normalize()
        >>> data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        >>> warped = norm.timewarp(data)
        >>> warped.shape
        (2, 3)
        """
        if not isinstance(segments, np.ndarray):
            raise TypeError("segments must be a np.ndarray.")
        if segments.ndim != 2:
            raise ValueError("Expected shape (N, window).")

        if mode not in ('local', 'global'):
            raise ValueError("mode must be 'local' or 'global'.")
        if mode == 'global':
            self._log("[timewarp] global mode is same as local placeholder...")

        out = segments.astype(np.float32).copy()
        n_seg = out.shape[0]
        self._log(f"[timewarp placeholder] mode={mode}, for {n_seg} segments.")
        return out

    def l2norm(
            self,
            segments: Annotated[np.ndarray,
            "Input segments of shape (N, window)"],
            mode: Annotated[str, "Mode: 'local' or 'global'"] = 'local',
            global_l2norm: Annotated[Optional[float],
            "Global L2 norm for 'global' mode"] = None
    ) -> Annotated[np.ndarray, "L2-normalized segments"]:
        """
        Perform L2 normalization on each segment.

        Parameters
        ----------
        segments : numpy.ndarray
            2D NumPy array of shape (N, window).
        mode : {'local', 'global'}, optional
            - 'local': Normalize each segment by its own L2 norm.
            - 'global': Normalize all segments by a global L2 norm.
        global_l2norm : float, optional
            Global L2 norm used for 'global' mode. Must be > 1e-12 if used.

        Returns
        -------
        numpy.ndarray
            L2-normalized segments of the same shape as the input.

        Raises
        ------
        TypeError
            If segments is not a NumPy array.
        ValueError
            If segments is not 2-dimensional or mode is invalid.
        ValueError
            If global_l2norm is not provided or <= 1e-12 in 'global' mode.

        Examples
        --------
        >>> norm = Normalize()
        >>> data = np.array([[3, 4], [0, 5]], dtype=np.float32)
        >>> norm.l2norm(data)
        array([[0.6       , 0.8       ],
               [0.        , 1.        ]], dtype=float32)

        >>> g_l2 = float(np.sqrt(np.sum(data * data)))
        >>> norm.l2norm(data, mode='global', global_l2norm=g_l2)
        array([[0.42426407, 0.56568545],
               [0.        , 0.70710677]], dtype=float32)
        """
        if not isinstance(segments, np.ndarray):
            raise TypeError("segments must be a np.ndarray.")
        if segments.ndim != 2:
            raise ValueError("Expected shape (N, window).")

        out = segments.astype(np.float32).copy()
        n_seg = out.shape[0]

        if mode == 'local':
            for i in range(n_seg):
                row = out[i]
                energy = np.sum(row * row)
                if energy > 1e-12:
                    out[i] = row / np.sqrt(energy)
            self._log(f"[l2norm local] Applied to {n_seg} segments.")

        elif mode == 'global':
            if global_l2norm is None or global_l2norm < 1e-12:
                raise ValueError("Need global_l2norm > 1e-12 for global mode.")
            out /= global_l2norm
            self._log(
                f"[l2norm global] shape={out.shape}, factor=1/{global_l2norm}"
            )

        else:
            raise ValueError("mode must be 'local' or 'global'.")

        return out


def pad_to_length(
        row: Annotated[np.ndarray, "1D array to be padded or truncated"],
        length: Annotated[int, "Target length"]
) -> Annotated[np.ndarray, "Resulting array of shape (length,)"]:
    """
    Pad or truncate a 1D array to a specified length.

    Parameters
    ----------
    row : numpy.ndarray
        1D array to be processed.
    length : int
        Target length for the resulting array.

    Returns
    -------
    numpy.ndarray
        A 1D array of the specified length.

    Examples
    --------
    >>> data = np.array([1, 2, 3], dtype=np.float32)
    >>> pad_to_length(data, 5)
    array([1., 2., 3., 0., 0.], dtype=float32)

    >>> pad_to_length(data, 2)
    array([1., 2.], dtype=float32)
    """
    if not isinstance(row, np.ndarray):
        raise TypeError("row must be a NumPy array.")
    if row.ndim != 1:
        raise ValueError("Expected a 1D array.")
    if not isinstance(length, int):
        raise TypeError("length must be an integer.")

    r = row
    if len(r) < length:
        r = np.pad(r, (0, length - len(r)))
    else:
        r = r[:length]
    return r.astype(np.float32)


if __name__ == "__main__":
    arr1 = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    arr2 = np.linspace(0, 10, 10, dtype=np.float32)
    arr3 = np.linspace(5, -5, 30, dtype=np.float32)

    max_len_any = 50

    all_data_list = [
        pad_to_length(arr1, max_len_any),
        pad_to_length(arr2, max_len_any),
        pad_to_length(arr3, max_len_any),
    ]
    all_data = np.stack(all_data_list)
    print("All data shape:", all_data.shape)

    normalize = Normalize()

    pk_local = normalize.peak(all_data)
    print("[peak local]:\n", pk_local)

    test_global_absmax = np.max(np.abs(all_data))
    pk_global = normalize.peak(all_data, mode='global',
                               global_absmax=test_global_absmax)
    print("[peak global]:\n", pk_global)

    zs_local = normalize.zscore(all_data)
    print("[zscore local]:\n", zs_local)

    test_global_mean = float(np.mean(all_data))
    test_global_std = float(np.std(all_data))
    zs_global = normalize.zscore(all_data, mode='global',
                                 global_mean=test_global_mean,
                                 global_std=test_global_std)
    print("[zscore global]:\n", zs_global)

    test_rg_local = normalize.rankgauss(all_data)
    print("[rankgauss local]:\n", test_rg_local)

    test_rg_global = normalize.rankgauss(all_data, mode='global')
    print("[rankgauss global]:\n", test_rg_global)
