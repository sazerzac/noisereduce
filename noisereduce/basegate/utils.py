from typing import Optional, Union, Tuple

import numpy as np


def std_mean(
        arr: np.ndarray,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the standard deviation and mean over the dimensions specified by dim.

    Arguments:
        arr (ndarray): Input array.
        axis (int or tuple of ints, optional): Axis or axes along which the standard deviation and mean are computed.
                                               Default is None, which computes over the entire array.
        keepdims (bool, optional): Whether to keep the dimensions of the original array. Default is False.

    Returns:
        tuple: A tuple containing (standard deviation, mean).
    """
    mean = np.mean(arr, axis=axis, keepdims=keepdims)
    std = np.std(arr, axis=axis, keepdims=keepdims)
    return std, mean


def temperature_sigmoid(x: np.ndarray, x0: float, temp_coeff: float) -> np.ndarray:
    """
    Apply a sigmoid function with temperature scaling.

    Arguments:
        x (np.ndarray): Input tensor.
        x0 (float): Parameter that controls the threshold of the sigmoid.
        temp_coeff (float): Parameter that controls the slope of the sigmoid.

    Returns:
        np.ndarray: Output tensor after applying the sigmoid with temperature scaling.
    """
    return 1 / (1 + np.exp(-(x - x0) / temp_coeff))


def amp_to_db(x: np.ndarray, eps: float = np.finfo(np.float64).eps, top_db: float = 80.0) -> np.ndarray:
    """
    Convert the input tensor from amplitude to decibel scale.

    This function transforms an amplitude input `x` (e.g., audio signal or magnitude spectrum)
    into a decibel (dB) scale.
    It applies the formula:

        dB = 20 * log10(abs(x) + eps)

    where `eps` is a small constant added to avoid log of zero. The function also limits the maximum
    value of the decibel output to `top_db` decibels below the maximum value across the last axis of `x`.

    Arguments:
        x (np.ndarray): The input array, typically representing amplitude values (e.g., audio signal or spectrum).
        eps (float, optional): A small constant added to the amplitude to prevent log of zero. Default is the smallest positive float64.
        top_db (float, optional): The threshold value in decibels. Any values above this threshold will be clipped. Default is 80.0 dB.

    Returns:
        np.ndarray: The input array converted to decibels, with values clipped at `top_db` below the maximum value.
    """
    x_db = 20 * np.log10(np.abs(x) + eps)
    return np.maximum(x_db, np.max(x_db, axis=-1, keepdims=True) - top_db)
