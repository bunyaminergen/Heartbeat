# Standard library imports
from typing import Any, Annotated

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


class OneDCNN(nn.Module):
    """
    A one-dimensional CNN model with three convolutional blocks followed
    by two max-pooling operations and two fully connected layers.

    This class is designed for classification tasks with 1D inputs of
    fixed length. Each block applies convolution, batch normalization,
    and GELU activation. After some blocks, max pooling is applied to
    reduce the spatial dimension.

    Parameters
    ----------
    config : object
        Configuration object containing:
        seq_length : int
            Input sequence length.
        num_classes : int
            Number of output classes.
        conv1_out : int
            Output channels in block1.
        conv2_out : int
            Output channels in block2.
        conv3_out : int
            Output channels in block3.
        kernel_size : int
            Convolution kernel size.
        hidden_dim : int
            Dimensionality of the hidden layer.

    Attributes
    ----------
    conv1a : nn.Conv1d
        First convolution in the first block.
    bn1a : nn.BatchNorm1d
        Batch normalization for `conv1a`.
    conv1b : nn.Conv1d
        Second convolution in the first block.
    bn1b : nn.BatchNorm1d
        Batch normalization for `conv1b`.
    conv2a : nn.Conv1d
        First convolution in the second block.
    bn2a : nn.BatchNorm1d
        Batch normalization for `conv2a`.
    conv2b : nn.Conv1d
        Second convolution in the second block.
    bn2b : nn.BatchNorm1d
        Batch normalization for `conv2b`.
    conv3a : nn.Conv1d
        First convolution in the third block.
    bn3a : nn.BatchNorm1d
        Batch normalization for `conv3a`.
    conv3b : nn.Conv1d
        Second convolution in the third block.
    bn3b : nn.BatchNorm1d
        Batch normalization for `conv3b`.
    pool1 : nn.MaxPool1d
        Max-pooling layer after second block.
    pool2 : nn.MaxPool1d
        Max-pooling layer after third block.
    fc1 : nn.Linear
        First fully connected layer.
    fc2 : nn.Linear
        Second fully connected layer (output layer).
    gelu : nn.GELU
        GELU activation function.

    Methods
    -------
    forward(x)
        Forward pass of the network.

    Examples
    --------
    >>> import torch
    >>> config = type('Config', (object,), {
    ...     'seq_length': 128,
    ...     'num_classes': 10,
    ...     'conv1_out': 16,
    ...     'conv2_out': 32,
    ...     'conv3_out': 64,
    ...     'kernel_size': 3,
    ...     'hidden_dim': 128,
    ... })()
    >>> model = OneDCNN(config)
    >>> input_test = torch.randn(2, 1, config.seq_length)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([2, 10])
    """

    def __init__(
            self,
            config: Annotated[Any, "Configuration object with model hyperparameters"]
    ) -> None:
        """
        Initialize the OneDCNN model.

        Parameters
        ----------
        config : object
            Configuration object containing model hyperparameters.
        """
        super(OneDCNN, self).__init__()

        if not isinstance(config.seq_length, int):
            raise TypeError("Expected config.seq_length to be an int.")
        if not isinstance(config.classes, int):
            raise TypeError("Expected config.classes to be an int.")
        if not isinstance(config.convolutional[0].channels, int):
            raise TypeError("Expected config.convolutional[0] to be an int.")
        if not isinstance(config.convolutional[1].channels, int):
            raise TypeError("Expected config.convolutional[1] to be an int.")
        if not isinstance(config.convolutional[2].channels, int):
            raise TypeError("Expected config.convolutional[2] to be an int.")
        if not isinstance(config.kernel, int):
            raise TypeError("Expected config.kernel_size to be an int.")
        if not isinstance(config.hidden_dim, int):
            raise TypeError("Expected config.hidden_dim to be an int.")

        seq_length = config.seq_length
        num_classes = config.classes
        conv1_out = config.convolutional[0].channels
        conv2_out = config.convolutional[1].channels
        conv3_out = config.convolutional[2].channels
        k_size = config.kernel
        hidden_dim = config.hidden_dim

        self.conv1a = nn.Conv1d(
            in_channels=1, out_channels=conv1_out,
            kernel_size=k_size, padding=k_size // 2
        )
        self.bn1a = nn.BatchNorm1d(conv1_out)
        self.conv1b = nn.Conv1d(
            in_channels=conv1_out, out_channels=conv1_out,
            kernel_size=k_size, padding=k_size // 2
        )
        self.bn1b = nn.BatchNorm1d(conv1_out)

        self.conv2a = nn.Conv1d(
            in_channels=conv1_out, out_channels=conv2_out,
            kernel_size=k_size, padding=k_size // 2
        )
        self.bn2a = nn.BatchNorm1d(conv2_out)
        self.conv2b = nn.Conv1d(
            in_channels=conv2_out, out_channels=conv2_out,
            kernel_size=k_size, padding=k_size // 2
        )
        self.bn2b = nn.BatchNorm1d(conv2_out)

        self.conv3a = nn.Conv1d(
            in_channels=conv2_out, out_channels=conv3_out,
            kernel_size=k_size, padding=k_size // 2
        )
        self.bn3a = nn.BatchNorm1d(conv3_out)
        self.conv3b = nn.Conv1d(
            in_channels=conv3_out, out_channels=conv3_out,
            kernel_size=k_size, padding=k_size // 2
        )
        self.bn3b = nn.BatchNorm1d(conv3_out)

        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2)

        flatten_dim = conv3_out * (seq_length // 4)
        self.fc1 = nn.Linear(flatten_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.gelu = nn.GELU()

    def forward(
            self,
            x: Annotated[torch.Tensor, "Input tensor of shape (batch, 1, seq_length)"]
    ) -> Annotated[torch.Tensor, "Output logits of shape (batch, num_classes)"]:
        """
        Forward pass of the OneDCNN model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, seq_length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, num_classes).

        Examples
        --------
        >>> import torch
        >>> config = type('Config', (object,), {
        ...     'seq_length': 128,
        ...     'num_classes': 10,
        ...     'conv1_out': 16,
        ...     'conv2_out': 32,
        ...     'conv3_out': 64,
        ...     'kernel_size': 3,
        ...     'hidden_dim': 128,
        ... })()
        >>> model = OneDCNN(config)
        >>> input_test = torch.randn(2, 1, config.seq_length)
        >>> outputs = model(inputs)
        >>> outputs.shape
        torch.Size([2, 10])
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected x to be a torch.Tensor.")
        if x.dim() != 3:
            raise ValueError(
                "Expected x to have 3 dimensions (batch, channels, length)."
            )

        x = self.conv1a(x)
        x = self.bn1a(x)
        x = self.gelu(x)

        x = self.conv1b(x)
        x = self.bn1b(x)
        x = self.gelu(x)

        x = self.conv2a(x)
        x = self.bn2a(x)
        x = self.gelu(x)

        x = self.conv2b(x)
        x = self.bn2b(x)
        x = self.gelu(x)
        x = self.pool1(x)

        x = self.conv3a(x)
        x = self.bn3a(x)
        x = self.gelu(x)

        x = self.conv3b(x)
        x = self.bn3b(x)
        x = self.gelu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = self.gelu(self.fc1(x))
        x = self.fc2(x)

        return x


class AdvancedOneDCNN(nn.Module):
    """
    An advanced one-dimensional CNN model with skip connections
    and a TransformerEncoder block.

    This class applies convolutional blocks with skip connections,
    followed by max pooling and a Transformer encoder layer to
    process temporal/spatial features. Finally, fully connected
    layers produce classification outputs.

    Parameters
    ----------
    config : object
        Configuration object containing:
        seq_length : int
            Input sequence length.
        num_classes : int
            Number of output classes.
        conv1_out : int
            Output channels in block1.
        conv2_out : int
            Output channels in block2.
        conv3_out : int
            Output channels in block3.
        kernel_size : int
            Convolution kernel size.
        hidden_dim : int
            Dimensionality of the hidden layer.

    Attributes
    ----------
    seq_length : int
        Sequence length from `config`.
    num_classes : int
        Number of output classes.
    block1 : nn.Sequential
        First convolutional block.
    shortcut1 : nn.Conv1d
        Skip connection for block1.
    block2 : nn.Sequential
        Second convolutional block with dilation.
    shortcut2 : nn.Conv1d
        Skip connection for block2.
    block3 : nn.Sequential
        Third convolutional block.
    shortcut3 : nn.Conv1d
        Skip connection for block3.
    pool1 : nn.MaxPool1d
        Max-pooling after block2.
    pool2 : nn.MaxPool1d
        Max-pooling after block3.
    transformer : nn.TransformerEncoder
        Transformer encoder layer.
    fc1 : nn.Linear
        Fully connected layer.
    fc2 : nn.Linear
        Output layer.
    gelu : nn.GELU
        GELU activation.

    Methods
    -------
    forward(x)
        Forward pass of the network.

    Examples
    --------
    >>> import torch
    >>> config = type('Config', (object,), {
    ...     'seq_length': 128,
    ...     'num_classes': 10,
    ...     'conv1_out': 16,
    ...     'conv2_out': 32,
    ...     'conv3_out': 64,
    ...     'kernel_size': 3,
    ...     'hidden_dim': 128,
    ... })()
    >>> model = AdvancedOneDCNN(config)
    >>> input_test = torch.randn(2, 1, config.seq_length)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([2, 10])
    """

    def __init__(
            self,
            config: Annotated[Any, "Configuration object with model hyperparameters"]
    ) -> None:
        """
        Initialize the AdvancedOneDCNN model.

        Parameters
        ----------
        config : object
            Configuration object containing model hyperparameters.
        """
        super(AdvancedOneDCNN, self).__init__()

        if not isinstance(config.seq_length, int):
            raise TypeError("Expected config.seq_length to be an int.")
        if not isinstance(config.classes, int):
            raise TypeError("Expected config.classes to be an int.")
        if not isinstance(config.convolutional[0].channels, int):
            raise TypeError("Expected config.convolutional[0] to be an int.")
        if not isinstance(config.convolutional[1].channels, int):
            raise TypeError("Expected config.convolutional[1] to be an int.")
        if not isinstance(config.convolutional[2].channels, int):
            raise TypeError("Expected config.convolutional[2] to be an int.")
        if not isinstance(config.kernel, int):
            raise TypeError("Expected config.kernel_size to be an int.")
        if not isinstance(config.hidden_dim, int):
            raise TypeError("Expected config.hidden_dim to be an int.")

        self.seq_length = config.seq_length
        self.num_classes = config.classes

        conv1_out = config.convolutional[0].channels
        conv2_out = config.convolutional[1].channels
        conv3_out = config.convolutional[2].channels
        k_size = config.kernel

        self.block1 = nn.Sequential(
            nn.Conv1d(1, conv1_out, kernel_size=k_size,
                      padding=k_size // 2),
            nn.BatchNorm1d(conv1_out),
            nn.GELU(),
            nn.Conv1d(conv1_out, conv1_out, kernel_size=k_size,
                      padding=k_size // 2),
            nn.BatchNorm1d(conv1_out),
            nn.GELU(),
        )
        self.shortcut1 = nn.Conv1d(1, conv1_out, kernel_size=1)

        self.block2 = nn.Sequential(
            nn.Conv1d(conv1_out, conv2_out, kernel_size=3,
                      padding=2, dilation=2),
            nn.BatchNorm1d(conv2_out),
            nn.GELU(),
            nn.Conv1d(conv2_out, conv2_out, kernel_size=3,
                      padding=2, dilation=2),
            nn.BatchNorm1d(conv2_out),
            nn.GELU(),
        )
        self.shortcut2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=1)

        self.block3 = nn.Sequential(
            nn.Conv1d(conv2_out, conv3_out, kernel_size=k_size,
                      padding=k_size // 2),
            nn.BatchNorm1d(conv3_out),
            nn.GELU(),
            nn.Conv1d(conv3_out, conv3_out, kernel_size=k_size,
                      padding=k_size // 2),
            nn.BatchNorm1d(conv3_out),
            nn.GELU(),
        )
        self.shortcut3 = nn.Conv1d(conv2_out, conv3_out, kernel_size=1)

        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conv3_out,
            nhead=4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        flatten_dim = conv3_out * (self.seq_length // 4)
        self.fc1 = nn.Linear(flatten_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, self.num_classes)

        self.gelu = nn.GELU()

    def forward(
            self,
            x: Annotated[torch.Tensor, "Input tensor of shape (batch, 1, seq_length)"]
    ) -> Annotated[torch.Tensor, "Output logits of shape (batch, num_classes)"]:
        """
        Forward pass of the AdvancedOneDCNN model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, seq_length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, num_classes).

        Examples
        --------
        >>> import torch
        >>> config = type('Config', (object,), {
        ...     'seq_length': 128,
        ...     'num_classes': 10,
        ...     'conv1_out': 16,
        ...     'conv2_out': 32,
        ...     'conv3_out': 64,
        ...     'kernel_size': 3,
        ...     'hidden_dim': 128,
        ... })()
        >>> model = AdvancedOneDCNN(config)
        >>> input_test = torch.randn(2, 1, config.seq_length)
        >>> outputs = model(inputs)
        >>> outputs.shape
        torch.Size([2, 10])
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected x to be a torch.Tensor.")
        if x.dim() != 3:
            raise ValueError(
                "Expected x to have 3 dimensions (batch, channels, length)."
            )

        out1 = self.block1(x)
        out1 = out1 + self.shortcut1(x)
        out1 = self.gelu(out1)

        out2 = self.block2(out1)
        out2 = out2 + self.shortcut2(out1)
        out2 = self.gelu(out2)
        out2 = self.pool1(out2)

        out3 = self.block3(out2)
        out3 = out3 + self.shortcut3(out2)
        out3 = self.gelu(out3)
        out3 = self.pool2(out3)

        out3 = out3.permute(0, 2, 1)
        out4 = self.transformer(out3)
        out4 = out4.permute(0, 2, 1)

        out4_flat = out4.reshape(out4.size(0), -1)

        out = self.gelu(self.fc1(out4_flat))
        out = self.fc2(out)
        return out


class SelfONNBlockOneD(nn.Module):
    """
    A basic Self-ONN block for 1D data that combines linear, squared, and
    square-root transformations, each learned with separate convolutional
    filters.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int, optional
        Convolution kernel size. Default is 3.

    Attributes
    ----------
    conv_linear : nn.Conv1d
        Convolution for the linear component.
    conv_square : nn.Conv1d
        Convolution for the squared component.
    conv_sqrt : nn.Conv1d
        Convolution for the square-root component.
    alpha_linear : nn.Parameter
        Learnable weight for the linear component.
    alpha_square : nn.Parameter
        Learnable weight for the squared component.
    alpha_sqrt : nn.Parameter
        Learnable weight for the square-root component.
    bn : nn.BatchNorm1d
        Batch normalization layer.

    Methods
    -------
    forward(x)
        Forward pass combining linear, square, and root transformations.

    Examples
    --------
    >>> import torch
    >>> block = SelfONNBlockOneD(in_channels=1, out_channels=4)
    >>> input_test = torch.randn(2, 1, 16)
    >>> outputs = block(inputs)
    >>> outputs.shape
    torch.Size([2, 4, 16])
    """

    def __init__(
            self,
            in_channels: Annotated[int, "Number of input channels"],
            out_channels: Annotated[int, "Number of output channels"],
            kernel_size: Annotated[int, "Convolution kernel size"] = 3
    ) -> None:
        """
        Initialize the SelfONNBlockOneD block.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int, optional
            Convolution kernel size. Default is 3.
        """
        super(SelfONNBlockOneD, self).__init__()

        if not isinstance(in_channels, int):
            raise TypeError("Expected in_channels to be an int.")
        if not isinstance(out_channels, int):
            raise TypeError("Expected out_channels to be an int.")
        if not isinstance(kernel_size, int):
            raise TypeError("Expected kernel_size to be an int.")

        self.conv_linear = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, padding=kernel_size // 2
        )
        self.conv_square = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, padding=kernel_size // 2
        )
        self.conv_sqrt = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, padding=kernel_size // 2
        )

        self.alpha_linear = nn.Parameter(torch.ones(out_channels))
        self.alpha_square = nn.Parameter(torch.ones(out_channels))
        self.alpha_sqrt = nn.Parameter(torch.ones(out_channels))

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(
            self,
            x: Annotated[torch.Tensor, "Input tensor of shape (batch, in_channels, length)"]
    ) -> Annotated[torch.Tensor, "Output tensor of shape (batch, out_channels, length)"]:
        """
        Forward pass of the SelfONNBlockOneD.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_channels, length).

        Examples
        --------
        >>> import torch
        >>> block = SelfONNBlockOneD(in_channels=1, out_channels=4)
        >>> input_test = torch.randn(2, 1, 16)
        >>> outputs = block(inputs)
        >>> outputs.shape
        torch.Size([2, 4, 16])
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected x to be a torch.Tensor.")
        if x.dim() != 3:
            raise ValueError("Expected x to have 3 dimensions.")

        out_lin = self.conv_linear(x)
        out_sq = self.conv_square(x)
        out_rt = self.conv_sqrt(x)

        out = (
                self.alpha_linear.view(1, -1, 1) * out_lin
                + self.alpha_square.view(1, -1, 1) * (out_sq ** 2)
                + self.alpha_sqrt.view(1, -1, 1)
                * torch.sqrt(torch.abs(out_rt) + 1e-8)
        )
        out = self.bn(out)
        return out


class OneDSelfONN(nn.Module):
    """
    A 3-block Self-ONN model (each block is SelfONNBlockOneD) with
    two max-pool operations. This reduces the sequence length by a
    factor of 4 overall. The channel architecture progresses as:
    conv1_out -> conv2_out -> conv3_out.

    Parameters
    ----------
    config : object
        Configuration object containing:
        seq_length : int
            Input sequence length.
        num_classes : int
            Number of output classes.
        conv1_out : int
            Output channels in block1.
        conv2_out : int
            Output channels in block2.
        conv3_out : int
            Output channels in block3.
        kernel_size : int
            Convolution kernel size.
        hidden_dim : int
            Dimensionality of the hidden layer.

    Attributes
    ----------
    block1 : SelfONNBlockOneD
        First Self-ONN block.
    act1 : nn.GELU
        GELU activation for first block.
    block2 : SelfONNBlockOneD
        Second Self-ONN block.
    act2 : nn.GELU
        GELU activation for second block.
    block3 : SelfONNBlockOneD
        Third Self-ONN block.
    act3 : nn.GELU
        GELU activation for third block.
    pool1 : nn.MaxPool1d
        First max-pooling layer.
    pool2 : nn.MaxPool1d
        Second max-pooling layer.
    fc1 : nn.Linear
        Fully connected layer.
    fc2 : nn.Linear
        Output layer.
    gelu : nn.GELU
        GELU activation function.

    Methods
    -------
    forward(x)
        Forward pass of the network.

    Examples
    --------
    >>> import torch
    >>> config = type('Config', (object,), {
    ...     'seq_length': 128,
    ...     'num_classes': 10,
    ...     'conv1_out': 16,
    ...     'conv2_out': 32,
    ...     'conv3_out': 64,
    ...     'kernel_size': 3,
    ...     'hidden_dim': 128,
    ... })()
    >>> model = OneDSelfONN(config)
    >>> input_test = torch.randn(2, 1, config.seq_length)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([2, 10])
    """

    def __init__(
            self,
            config: Annotated[Any, "Configuration object with model hyperparameters"]
    ) -> None:
        """
        Initialize the OneDSelfONN model.

        Parameters
        ----------
        config : object
            Configuration object containing model hyperparameters.
        """
        super(OneDSelfONN, self).__init__()

        if not isinstance(config.seq_length, int):
            raise TypeError("Expected config.seq_length to be an int.")
        if not isinstance(config.classes, int):
            raise TypeError("Expected config.classes to be an int.")
        if not isinstance(config.convolutional[0].channels, int):
            raise TypeError("Expected config.convolutional[0] to be an int.")
        if not isinstance(config.convolutional[1].channels, int):
            raise TypeError("Expected config.convolutional[1] to be an int.")
        if not isinstance(config.convolutional[2].channels, int):
            raise TypeError("Expected config.convolutional[2] to be an int.")
        if not isinstance(config.kernel, int):
            raise TypeError("Expected config.kernel_size to be an int.")
        if not isinstance(config.hidden_dim, int):
            raise TypeError("Expected config.hidden_dim to be an int.")

        seq_length = config.seq_length
        num_classes = config.classes
        conv1_out = config.convolutional[0].channels
        conv2_out = config.convolutional[1].channels
        conv3_out = config.convolutional[2].channels
        k_size = config.kernel
        hidden_dim = config.hidden_dim

        self.block1 = SelfONNBlockOneD(
            in_channels=1,
            out_channels=conv1_out,
            kernel_size=k_size
        )
        self.act1 = nn.GELU()

        self.block2 = SelfONNBlockOneD(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=k_size
        )
        self.act2 = nn.GELU()

        self.block3 = SelfONNBlockOneD(
            in_channels=conv2_out,
            out_channels=conv3_out,
            kernel_size=k_size
        )
        self.act3 = nn.GELU()

        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2)

        flatten_dim = conv3_out * (seq_length // 4)
        self.fc1 = nn.Linear(flatten_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.gelu = nn.GELU()

    def forward(
            self,
            x: Annotated[torch.Tensor, "Input of shape (batch, 1, seq_length)"]
    ) -> Annotated[torch.Tensor, "Output logits of shape (batch, num_classes)"]:
        """
        Forward pass of the OneDSelfONN model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, seq_length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, num_classes).

        Examples
        --------
        >>> import torch
        >>> config = type('Config', (object,), {
        ...     'seq_length': 128,
        ...     'num_classes': 10,
        ...     'conv1_out': 16,
        ...     'conv2_out': 32,
        ...     'conv3_out': 64,
        ...     'kernel_size': 3,
        ...     'hidden_dim': 128,
        ... })()
        >>> model = OneDSelfONN(config)
        >>> input_test = torch.randn(2, 1, config.seq_length)
        >>> outputs = model(inputs)
        >>> outputs.shape
        torch.Size([2, 10])
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected x to be a torch.Tensor.")
        if x.dim() != 3:
            raise ValueError("Expected x to have 3 dimensions.")

        x = self.block1(x)
        x = self.act1(x)

        x = self.block2(x)
        x = self.act2(x)
        x = self.pool1(x)

        x = self.block3(x)
        x = self.act3(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)

        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class SelfONNBlockOneDAdvanced(nn.Module):
    """
    An advanced Self-ONN block that includes gating, exponent (x^p),
    and an optional skip connection.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int, optional
        Convolution kernel size. Default is 3.
    use_skip : bool, optional
        Whether to include a skip connection. Default is True.

    Attributes
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_skip : bool
        Flag indicating if skip connection is used.
    conv_linear : nn.Conv1d
        Convolution for the linear component.
    conv_square : nn.Conv1d
        Convolution for the squared component.
    conv_sqrt : nn.Conv1d
        Convolution for the sqrt component.
    conv_exponent : nn.Conv1d
        Convolution for the exponent (x^p) component.
    p : nn.Parameter
        Exponent parameter.
    gate_fc1 : nn.Linear
        First linear layer for gating.
    gate_fc2 : nn.Linear
        Second linear layer for gating.
    skip_conv : nn.Conv1d or None
        Skip connection if in_channels != out_channels and use_skip is True.
    bn : nn.BatchNorm1d
        Batch normalization.

    Methods
    -------
    forward(x)
        Forward pass combining linear, square, sqrt, and exponent terms,
        modulated by gating and skip connections.

    Examples
    --------
    >>> import torch
    >>> block = SelfONNBlockOneDAdvanced(
    ...     in_channels=1, out_channels=4,
    ... )
    >>> input_test = torch.randn(2, 1, 16)
    >>> outputs = block(inputs)
    >>> outputs.shape
    torch.Size([2, 4, 16])
    """

    def __init__(
            self,
            in_channels: Annotated[int, "Number of input channels"],
            out_channels: Annotated[int, "Number of output channels"],
            kernel_size: Annotated[int, "Convolution kernel size"] = 3,
            use_skip: Annotated[bool, "Whether to use skip connection"] = True
    ) -> None:
        """
        Initialize the advanced Self-ONN block.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int, optional
            Convolution kernel size. Default is 3.
        use_skip : bool, optional
            Whether to use a skip connection. Default is True.
        """
        super(SelfONNBlockOneDAdvanced, self).__init__()

        if not isinstance(in_channels, int):
            raise TypeError("Expected in_channels to be an int.")
        if not isinstance(out_channels, int):
            raise TypeError("Expected out_channels to be an int.")
        if not isinstance(kernel_size, int):
            raise TypeError("Expected kernel_size to be an int.")
        if not isinstance(use_skip, bool):
            raise TypeError("Expected use_skip to be a bool.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_skip = use_skip

        self.conv_linear = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, padding=kernel_size // 2
        )
        self.conv_square = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, padding=kernel_size // 2
        )
        self.conv_sqrt = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, padding=kernel_size // 2
        )
        self.conv_exponent = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, padding=kernel_size // 2
        )

        self.p = nn.Parameter(torch.tensor(1.5, dtype=torch.float32),
                              requires_grad=True)

        reduction = max(4, out_channels // 8)
        self.gate_fc1 = nn.Linear(out_channels, reduction)
        self.gate_fc2 = nn.Linear(reduction, 4 * out_channels)

        if self.use_skip and (in_channels != out_channels):
            self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = None

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(
            self,
            x: Annotated[torch.Tensor, "Input tensor of shape (batch, in_channels, length)"]
    ) -> Annotated[torch.Tensor, "Output tensor of shape (batch, out_channels, length)"]:
        """
        Forward pass of the advanced Self-ONN block.

        Combines linear, square, sqrt, and exponent terms, then
        applies a squeeze-and-excitation-style gating mechanism
        to reweight each component. A skip connection is added
        if `use_skip` is True.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, out_channels, length).

        Examples
        --------
        >>> import torch
        >>> block = SelfONNBlockOneDAdvanced(
        ...     in_channels=1, out_channels=4
        ... )
        >>> input_test = torch.randn(2, 1, 16)
        >>> outputs = block(inputs)
        >>> outputs.shape
        torch.Size([2, 4, 16])
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected x to be a torch.Tensor.")
        if x.dim() != 3:
            raise ValueError("Expected x to have 3 dimensions.")

        out_lin = self.conv_linear(x)
        out_sq = self.conv_square(x)
        out_rt = self.conv_sqrt(x)
        out_exp = self.conv_exponent(x)

        # noinspection PyAugmentAssignment
        out_sq = out_sq ** 2
        out_rt = torch.sqrt(torch.abs(out_rt) + 1e-8)

        sign_exp = torch.sign(out_exp)
        mag_exp = torch.abs(out_exp)
        out_exp = sign_exp * (mag_exp ** self.p)

        combined_feats = out_lin + out_sq + out_rt + out_exp
        se_vector = F.adaptive_avg_pool1d(combined_feats, 1).squeeze(-1)

        gate = self.gate_fc1(se_vector)
        gate = F.gelu(gate)
        gate = self.gate_fc2(gate)
        gate = gate.view(-1, 4, self.out_channels)
        gate = F.softmax(gate, dim=1)

        # noinspection PyAugmentAssignment
        out_lin = out_lin * gate[:, 0, :].unsqueeze(-1)
        # noinspection PyAugmentAssignment
        out_sq = out_sq * gate[:, 1, :].unsqueeze(-1)
        # noinspection PyAugmentAssignment
        out_rt = out_rt * gate[:, 2, :].unsqueeze(-1)
        # noinspection PyAugmentAssignment
        out_exp = out_exp * gate[:, 3, :].unsqueeze(-1)

        out = out_lin + out_sq + out_rt + out_exp

        if self.use_skip:
            if self.skip_conv is not None:
                out = out + self.skip_conv(x)
            else:
                out += x

        out = self.bn(out)
        return out


class AdvancedOneDSelfONN(nn.Module):
    """
    An advanced 1D Self-ONN model with three blocks
    (SelfONNBlockOneDAdvanced). Each block is followed by a max-pool
    (size 2), which reduces the sequence length by a factor of 8 in
    total (2 * 2 * 2).

    Parameters
    ----------
    config : object
        Configuration object containing:
        seq_length : int
            Input sequence length.
        num_classes : int
            Number of output classes.
        conv1_out : int
            Output channels in block1.
        conv2_out : int
            Output channels in block2.
        conv3_out : int
            Output channels in block3.
        kernel_size : int
            Convolution kernel size.
        hidden_dim : int
            Dimensionality of the hidden layer.

    Attributes
    ----------
    block1 : SelfONNBlockOneDAdvanced
        First advanced Self-ONN block.
    act1 : nn.GELU
        GELU activation after block1.
    pool1 : nn.MaxPool1d
        Max-pooling after block1.
    block2 : SelfONNBlockOneDAdvanced
        Second advanced Self-ONN block.
    act2 : nn.GELU
        GELU activation after block2.
    pool2 : nn.MaxPool1d
        Max-pooling after block2.
    block3 : SelfONNBlockOneDAdvanced
        Third advanced Self-ONN block.
    act3 : nn.GELU
        GELU activation after block3.
    pool3 : nn.MaxPool1d
        Max-pooling after block3.
    fc1 : nn.Linear
        Fully connected layer.
    fc2 : nn.Linear
        Output layer.
    gelu : nn.GELU
        GELU activation.

    Methods
    -------
    forward(x)
        Forward pass of the network.

    Examples
    --------
    >>> import torch
    >>> config = type('Config', (object,), {
    ...     'seq_length': 128,
    ...     'num_classes': 10,
    ...     'conv1_out': 16,
    ...     'conv2_out': 32,
    ...     'conv3_out': 64,
    ...     'kernel_size': 3,
    ...     'hidden_dim': 128,
    ... })()
    >>> model = AdvancedOneDSelfONN(config)
    >>> input_test = torch.randn(2, 1, config.seq_length)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([2, 10])
    """

    def __init__(
            self,
            config: Annotated[Any, "Configuration object with model hyperparameters"]
    ) -> None:
        """
        Initialize the AdvancedOneDSelfONN model.

        Parameters
        ----------
        config : object
            Configuration object containing model hyperparameters.
        """
        super(AdvancedOneDSelfONN, self).__init__()

        if not isinstance(config.seq_length, int):
            raise TypeError("Expected config.seq_length to be an int.")
        if not isinstance(config.classes, int):
            raise TypeError("Expected config.classes to be an int.")
        if not isinstance(config.convolutional[0].channels, int):
            raise TypeError("Expected config.convolutional[0] to be an int.")
        if not isinstance(config.convolutional[1].channels, int):
            raise TypeError("Expected config.convolutional[1] to be an int.")
        if not isinstance(config.convolutional[2].channels, int):
            raise TypeError("Expected config.convolutional[2] to be an int.")
        if not isinstance(config.kernel, int):
            raise TypeError("Expected config.kernel_size to be an int.")
        if not isinstance(config.hidden_dim, int):
            raise TypeError("Expected config.hidden_dim to be an int.")

        seq_length = config.seq_length
        num_classes = config.classes
        conv1_out = config.convolutional[0].channels
        conv2_out = config.convolutional[1].channels
        conv3_out = config.convolutional[2].channels
        k_size = config.kernel
        hidden_dim = config.hidden_dim

        self.block1 = SelfONNBlockOneDAdvanced(
            in_channels=1,
            out_channels=conv1_out,
            kernel_size=k_size,
        )
        self.act1 = nn.GELU()
        self.pool1 = nn.MaxPool1d(2)

        self.block2 = SelfONNBlockOneDAdvanced(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=k_size,
        )
        self.act2 = nn.GELU()
        self.pool2 = nn.MaxPool1d(2)

        self.block3 = SelfONNBlockOneDAdvanced(
            in_channels=conv2_out,
            out_channels=conv3_out,
            kernel_size=k_size,
        )
        self.act3 = nn.GELU()
        self.pool3 = nn.MaxPool1d(2)

        flatten_dim = conv3_out * (seq_length // 8)

        self.fc1 = nn.Linear(flatten_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.gelu = nn.GELU()

    def forward(
            self,
            x: Annotated[torch.Tensor, "Input of shape (batch, 1, seq_length)"]
    ) -> Annotated[torch.Tensor, "Output logits of shape (batch, num_classes)"]:
        """
        Forward pass of the AdvancedOneDSelfONN model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, seq_length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, num_classes).

        Examples
        --------
        >>> import torch
        >>> config = type('Config', (object,), {
        ...     'seq_length': 128,
        ...     'num_classes': 10,
        ...     'conv1_out': 16,
        ...     'conv2_out': 32,
        ...     'conv3_out': 64,
        ...     'kernel_size': 3,
        ...     'hidden_dim': 128,
        ... })()
        >>> model = AdvancedOneDSelfONN(config)
        >>> input_test = torch.randn(2, 1, config.seq_length)
        >>> outputs = model(inputs)
        >>> outputs.shape
        torch.Size([2, 10])
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected x to be a torch.Tensor.")
        if x.dim() != 3:
            raise ValueError("Expected x to have 3 dimensions.")

        x = self.block1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.act3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)

        x = self.gelu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    config_path = "config/config.yaml"
    configuration = OmegaConf.load(config_path)
    model_config = configuration.model

    inputs = torch.randn(2, 1, model_config.seq_length)

    model_1 = OneDCNN(model_config)
    y_1 = model_1(inputs)

    model_2 = OneDSelfONN(model_config)
    y_2 = model_2(inputs)

    model_3 = AdvancedOneDCNN(model_config)
    y_3 = model_3(inputs)

    model_4 = AdvancedOneDSelfONN(model_config)
    y_4 = model_4(inputs)

    print("OneDCNN output shape:", y_1.shape)
    print("OneDSelfONN output shape:", y_2.shape)
    print("AdvancedOneDCNN output shape:", y_3.shape)
    print("AdvancedOneDSelfONN output shape:", y_4.shape)
