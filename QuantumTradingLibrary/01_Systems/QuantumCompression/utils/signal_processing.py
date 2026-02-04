# QuantumTradingLibrary/signal_processing.py
#
# Author: DooDoo
# Date: 2026-01-19
#
# Description:
# This module contains classes and functions for advanced signal processing of financial time series.
# It includes wavelet and CEEMDAN denoising, compression-based regime detection, and validation utilities
# to prepare data for quantum machine learning models.

from typing import Dict, Optional, Tuple, List
import numpy as np
import pywt
from PyEMD import CEEMDAN

# Suppress PyEMD informational messages
import logging
logging.basicConfig(level=logging.WARNING)


class NoiseReducer:
    """
    A comprehensive pipeline for denoising financial time series data.
    Combines wavelet transforms and CEEMDAN for multi-resolution noise reduction.
    """

    def __init__(
        self,
        wavelet: str = 'db8',
        wavelet_level: int = 4,
        ceemdan_trials: int = 100
    ):
        """
        Initialize the denoiser with chosen parameters.

        Args:
            wavelet: Name of the mother wavelet (e.g., 'db8', 'sym8').
            wavelet_level: Decomposition level for the wavelet transform.
            ceemdan_trials: Number of trials for the CEEMDAN decomposition.
        """
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.ceemdan_trials = ceemdan_trials
        self.metrics = {}

    def wavelet_denoise(self, data: np.ndarray) -> np.ndarray:
        """
        Denoise data using wavelet transform with universal thresholding.

        Args:
            data: 1D numpy array of time series data.

        Returns:
            Denoised 1D numpy array.
        """
        coeffs = pywt.wavedec(data, self.wavelet, level=self.wavelet_level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))

        new_coeffs = [coeffs[0]]
        for c in coeffs[1:]:
            new_coeffs.append(pywt.threshold(c, threshold, mode='soft'))

        denoised = pywt.waverec(new_coeffs, self.wavelet)
        # Ensure output length matches input length
        return denoised[:len(data)]

    def midas_style_denoise(self, data: np.ndarray, wavelet='db4', level=4) -> np.ndarray:
        """
        Simple but effective denoising using a fixed wavelet and level.
        This is a good starting point for quick integration.
        """
        coeffs = pywt.wavedec(data, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))

        new_coeffs = [coeffs[0]]
        for c in coeffs[1:]:
            new_coeffs.append(pywt.threshold(c, threshold, mode='soft'))

        return pywt.waverec(new_coeffs, wavelet)[:len(data)]

    def ceemdan_denoise(
        self,
        data: np.ndarray,
        reconstruction_imfs: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Denoise data using Complete Ensemble EMD with Adaptive Noise.
        Separates data into IMFs, reconstructs signal from lower-frequency IMFs.

        Args:
            data: 1D numpy array of time series data.
            reconstruction_imfs: List of IMF indices to use for reconstruction.
                                 If None, uses a default heuristic.

        Returns:
            A tuple containing:
            - denoised_signal (np.ndarray): The reconstructed, denoised signal.
            - trend (np.ndarray): The long-term trend component (residue).
        """
        ceemdan = CEEMDAN(trials=self.ceemdan_trials)
        imfs = ceemdan(data)

        if reconstruction_imfs is None:
            # Heuristic: use the last ~1/3 of IMFs plus the residue for signal
            num_imfs_for_signal = max(2, int(len(imfs) / 3))
            signal_imfs = imfs[-num_imfs_for_signal:]
        else:
            signal_imfs = [imfs[i] for i in reconstruction_imfs if i < len(imfs)]

        denoised_signal = np.sum(signal_imfs, axis=0)
        trend = imfs[-1] # The residue is the long-term trend

        # The noise is the sum of the first few IMFs (high-frequency components)
        noise_imfs = imfs[:-num_imfs_for_signal]
        self.metrics['ceemdan_noise_components'] = noise_imfs

        return denoised_signal, trend

    def full_pipeline(
        self,
        data: np.ndarray,
        return_components: bool = False
    ) -> Dict | np.ndarray:
        """
        Execute the full denoising pipeline: Wavelet -> CEEMDAN.

        Args:
            data: 1D numpy array of raw time series data.
            return_components: If True, returns a dictionary of all intermediate stages.

        Returns:
            If return_components is False, returns the final denoised data array.
            If True, returns a dictionary with original, intermediate, and final data.
        """
        original = np.copy(data)

        # Stage 1: Wavelet denoising
        stage1 = self.wavelet_denoise(data)

        # Stage 2: CEEMDAN (optional, can be slow)
        try:
            stage2, trend = self.ceemdan_denoise(stage1)
        except Exception as e:
            print(f"CEEMDAN failed, using wavelet output: {e}")
            stage2 = stage1
            trend = stage1

        # Stage 3: Compression metrics (for regime detection) - moved to RegimeDetector
        # compression_metrics = self.compute_compression_metrics(stage2)

        # Calculate overall noise reduction
        self.metrics['total_variance_reduction'] = 1 - np.var(stage2) / np.var(original)
        self.metrics['snr_improvement_db'] = 10 * np.log10(np.var(original) / np.var(original - stage2))

        if return_components:
            return {
                'original': original,
                'wavelet_denoised': stage1,
                'ceemdan_denoised': stage2,
                'trend': trend,
                'noise_removed': original - stage2,
                'metrics': self.metrics
            }

        return stage2

    def prepare_for_quantum(
        self,
        data: np.ndarray,
        window_size: int = 64
    ) -> np.ndarray:
        """
        Prepare denoised data for quantum state encoding.
        Normalizes and shapes data for quantum circuit input.
        """
        denoised = self.full_pipeline(data)

        # Normalize to [0, 1] for quantum state amplitude encoding
        min_val, max_val = denoised.min(), denoised.max()
        normalized = (denoised - min_val) / (max_val - min_val + 1e-8)

        # Reshape into windows
        n_windows = len(normalized) // window_size
        windowed = normalized[:n_windows * window_size].reshape(n_windows, window_size)

        return windowed


class RegimeDetector:
    """
    Compression-based market regime detection.
    Uses NCD (Normalized Compression Distance) to find similar historical periods.
    """

    def __init__(self, algorithm: str = 'bz2', window_size: int = 100):
        """
        Initialize the detector.

        Args:
            algorithm: Compression algorithm to use ('bz2' or 'zlib').
            window_size: Size of the rolling window for complexity detection.
        """
        self.algorithm = algorithm
        self.window_size = window_size
        self.historical_regimes = []

    def _compress(self, data: np.ndarray) -> bytes:
        import bz2
        import zlib

        data_bytes = data.astype(np.float32).tobytes()

        if self.algorithm == 'bz2':
            return bz2.compress(data_bytes, compresslevel=9)
        else:
            return zlib.compress(data_bytes, level=9)

    def ncd(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the Normalized Compression Distance between two data arrays.
        """
        cx = len(self._compress(x))
        cy = len(self._compress(y))

        combined = np.concatenate([x, y])
        cxy = len(self._compress(combined))

        return (cxy - min(cx, cy)) / max(cx, cy)

    def add_regime(self, data: np.ndarray, label: str):
        """
        Add a labeled historical regime to the database.

        Args:
            data: The time series data for the regime.
            label: A human-readable label (e.g., 'high_volatility', 'bull_run').
        """
        self.historical_regimes.append({
            'data': data,
            'label': label,
            'compression_ratio': len(data.tobytes()) / len(self._compress(data))
        })

    def detect_regime(self, current_data: np.ndarray, top_k: int = 3) -> list:
        """
        Find the most similar historical regimes to the current data.
        """
        if not self.historical_regimes:
            return []

        similarities = []
        for regime in self.historical_regimes:
            ncd_score = self.ncd(current_data, regime['data'])
            similarities.append({
                'label': regime['label'],
                'ncd': ncd_score,
                'similarity': 1 - ncd_score
            })

        similarities.sort(key=lambda x: x['ncd'])
        return similarities[:top_k]

    def complexity_change_detector(
        self,
        data: np.ndarray,
        lookback: int = 10
    ) -> np.ndarray:
        """
        Detects regime changes by analyzing the z-score of compression ratio changes.
        Sharp spikes in the output indicate potential regime shifts.
        """
        ratios = []
        for i in range(self.window_size, len(data)):
            window = data[i-self.window_size:i]
            if len(window) == 0: continue
            ratio = len(window.tobytes()) / len(self._compress(window))
            ratios.append(ratio)

        ratios = np.array(ratios)

        # Z-score of compression ratio changes
        ratio_diff = np.diff(ratios)
        z_scores = (ratio_diff - ratio_diff.mean()) / (ratio_diff.std() + 1e-8)

        return z_scores


# ============================================================
# VALIDATION UTILITIES
# ============================================================

def validate_denoising_improvement(
    original: np.ndarray,
    denoised: np.ndarray,
    model_func,
    n_trials: int = 10
) -> Dict[str, float]:
    """
    Validate that denoising actually improves a given model's prediction accuracy.

    Args:
        original: Raw time series data.
        denoised: Denoised time series data.
        model_func: A function that takes data and returns a prediction accuracy score.
        n_trials: Number of validation trials to run.

    Returns:
        A dictionary containing mean accuracies and the calculated improvement.
    """
    original_scores = [model_func(original) for _ in range(n_trials)]
    denoised_scores = [model_func(denoised) for _ in range(n_trials)]

    return {
        'original_mean_accuracy': np.mean(original_scores),
        'denoised_mean_accuracy': np.mean(denoised_scores),
        'improvement': np.mean(denoised_scores) - np.mean(original_scores),
        'improvement_pct': (np.mean(denoised_scores) - np.mean(original_scores)) / np.mean(original_scores) * 100,
        'original_std': np.std(original_scores),
        'denoised_std': np.std(denoised_scores)
    }


def calculate_entropy_reduction(
    original: np.ndarray,
    denoised: np.ndarray,
    bins: int = 50
) -> Dict[str, float]:
    """
    Calculate the reduction in entropy resulting from denoising.
    Compares both the entropy of the data values and their returns.
    """
    from scipy.stats import entropy

    def approx_entropy(data, bins):
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]
        return entropy(hist)

    orig_entropy = approx_entropy(original, bins)
    denoised_entropy = approx_entropy(denoised, bins)

    orig_returns = np.diff(original)
    denoised_returns = np.diff(denoised)

    orig_returns_entropy = approx_entropy(orig_returns, bins)
    denoised_returns_entropy = approx_entropy(denoised_returns, bins)

    return {
        'original_entropy': orig_entropy,
        'denoised_entropy': denoised_entropy,
        'entropy_reduction': orig_entropy - denoised_entropy,
        'entropy_reduction_pct': (orig_entropy - denoised_entropy) / orig_entropy * 100,
        'original_returns_entropy': orig_returns_entropy,
        'denoised_returns_entropy': denoised_returns_entropy,
        'returns_entropy_reduction': orig_returns_entropy - denoised_returns_entropy
    }
