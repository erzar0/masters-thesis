from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from skimage.transform import resize
from tqdm import tqdm

try:
    from .ArtificialTrainDataGenerator import ArtificialTrainDataGenerator
except Exception as e:
    from ArtificialTrainDataGenerator import ArtificialTrainDataGenerator


class DataPreprocessor:
    """
    Provides static methods for preprocessing spectral data.
    """

    @staticmethod
    def processed_spectra(
        spectra: np.ndarray,
        peaks: np.ndarray,
        energies: np.ndarray,
        min_energy: float = 0,
        max_energy: float = 20,
        energy_margin: float = 1,
        target_length: int = ArtificialTrainDataGenerator.CHANNELS,
        polyfit_deg: int = 1,
        extrapolate: bool = True,
        discrete_transformation: bool = False,
        extrapolation_kwargs: dict | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process 3D spectra (X, Y, C) through calibration, cropping, resizing,
        extrapolation, edge masking, and normalization.
        """
        peaks = np.asarray(peaks, dtype=float)
        energies = np.asarray(energies, dtype=float)
        if peaks.size < polyfit_deg + 1:
            raise ValueError(f"Cannot fit degree {polyfit_deg} polynomial with {peaks.size} points.")

        try:
            coeffs = np.polyfit(peaks, energies, polyfit_deg)
            energy_range = np.poly1d(coeffs)(np.arange(spectra.shape[-1]))
        except (np.linalg.LinAlgError, ValueError) as e:
            raise ValueError(f"Energy calibration failed: {e}")

        mask = (energy_range >= min_energy) & (energy_range <= max_energy)
        if np.count_nonzero(mask) < 2:
            raise ValueError("Insufficient data within target energy range.")

        masked_energy = energy_range[mask]
        masked_spectra = spectra[:, :, mask]

        if discrete_transformation:
            return DataPreprocessor._apply_discrete_transformation(
                spectra, energy_range, min_energy, max_energy,
                target_length, energy_margin
            )

        step = masked_energy[1] - masked_energy[0]
        pad_left = max(0, int(np.floor((masked_energy[0] - min_energy) / step)))
        pad_right = max(0, int(np.ceil((max_energy - masked_energy[-1]) / step)))

        padded_spectra = np.pad(masked_spectra, ((0, 0), (0, 0), (pad_left, pad_right)), mode='constant')
        resized = resize(
            padded_spectra,
            (spectra.shape[0], spectra.shape[1], target_length),
            order=1,
            preserve_range=True,
            anti_aliasing=target_length < padded_spectra.shape[-1]
        )

        target_energy = np.linspace(min_energy, max_energy, target_length)

        if extrapolate:
            extrapolation_kwargs = extrapolation_kwargs or {}
            resized = DataPreprocessor._extrapolate_spectra(resized, target_energy, **extrapolation_kwargs)

        resized *= DataPreprocessor._edge_mask(target_energy, min_energy, max_energy, energy_margin)
        return target_energy, DataPreprocessor._normalize(resized)

    @staticmethod
    def _apply_discrete_transformation(
        spectra: np.ndarray,
        energy_range: np.ndarray,
        min_energy: float,
        max_energy: float,
        target_length: int,
        energy_margin: float
    ) -> tuple[np.ndarray, np.ndarray]:
        X, Y, C = spectra.shape
        output = np.zeros((X, Y, target_length))
        for i in range(X):
            for j in range(Y):
                spec = spectra[i, j]
                new_idx = np.clip(
                    np.round(energy_range * target_length / (max_energy - min_energy)).astype(int),
                    0, target_length - 1
                )
                output[i, j] = np.bincount(new_idx, weights=spec, minlength=target_length)
        margin_idx = int(energy_margin * target_length / (max_energy - min_energy))
        output[:, :, :margin_idx] = 0
        output[:, :, -margin_idx:] = 0
        return energy_range, output

    @staticmethod
    def processed_spectrum(
        spectrum: np.ndarray,
        peaks: np.ndarray,
        energies: np.ndarray,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        if spectrum.ndim != 1:
            raise ValueError("Input spectrum must be 1D.")
        reshaped = spectrum.reshape(1, 1, -1)
        energy_range, processed = DataPreprocessor.processed_spectra(reshaped, peaks, energies, **kwargs)
        return energy_range, processed.flatten()

    @staticmethod
    def _edge_mask(energy_range, min_energy, max_energy, margin):
        mask = (energy_range >= min_energy + margin) & (energy_range <= max_energy - margin)
        return mask.astype(float)

    @staticmethod
    def _normalize(data: np.ndarray) -> np.ndarray:
        min_val = np.min(data, axis=-1, keepdims=True)
        max_val = np.max(data, axis=-1, keepdims=True)
        range_val = np.clip(max_val - min_val, 1e-9, None)
        return (data - min_val) / range_val

    @staticmethod
    def _extrapolate_spectra(
        spectra: np.ndarray,
        energy_range: np.ndarray,
        extrapolation_function: str | callable = 'exponential',
        channels_for_extrapolation: int = 32,
        function_params: dict | None = None,
        curve_fit_kwargs: dict | None = None
    ) -> np.ndarray:
        function_map = {
            'exponential': lambda x, a, b: a * np.exp(b * x),
            'quadratic': lambda x, a, b, c: a * x**2 + b * x + c
        }

        func = function_map.get(extrapolation_function, extrapolation_function)
        if not callable(func):
            raise ValueError("Extrapolation function must be callable or one of: 'exponential', 'quadratic'")

        try:
            param_count = len(inspect.signature(func).parameters) - 1
        except Exception:
            raise ValueError("Could not inspect the extrapolation function.")

        curve_fit_kwargs = curve_fit_kwargs or {'maxfev': 10000}
        warnings.warn("Pixel-wise extrapolation may be slow.", RuntimeWarning)
        X, Y, C = spectra.shape
        extrapolated = spectra.copy()

        for i in range(X):
            for j in range(Y):
                spectrum = extrapolated[i, j]
                if not np.any(spectrum):
                    continue
                non_zero = np.nonzero(spectrum)[0]
                if non_zero.size == 0:
                    continue

                start_idx, end_idx = non_zero[0], non_zero[-1]

                for side, (start, end, assign_range) in enumerate([
                    (start_idx, min(start_idx + channels_for_extrapolation, end_idx + 1), slice(None, start_idx + 1)),
                    (max(0, end_idx - channels_for_extrapolation + 1), end_idx + 1, slice(end_idx, None))
                ]):
                    x = energy_range[start:end]
                    y = spectrum[start:end]
                    if x.size <= param_count:
                        continue
                    p0 = None
                    if function_params:
                        p0 = function_params.get('start' if side == 0 else 'end')
                    try:
                        popt, _ = curve_fit(func, x, y, p0=p0, **curve_fit_kwargs)
                        extrap = func(energy_range[assign_range], *popt)
                        extrap[extrap < 0] = 0
                        spectrum[assign_range] = extrap
                    except (RuntimeError, ValueError):
                        continue

        return extrapolated


if __name__ == '__main__':
    E_SHIFT = 2
    path = Path("data/objects/Blank_PcbCu/GEM_Cr_PBS1_Blank_PcbCu_ArCO2_75_25_Gain_35_Thr_35_HV_3640_Xray_50kV_04mA_part_001_GainCorr.pcap.dat.npz")
    e_xy = np.load(path, allow_pickle=True)["E_xy"].astype(float) / (1 << E_SHIFT)

    PEAKS = [1000, 1614]
    PEAK_ENERGIES = [5.088, 8.046]
    spectrum = e_xy[50, 50]

    _, processed = DataPreprocessor.processed_spectrum(
        spectrum, PEAKS, PEAK_ENERGIES,
        discrete_transformation=True,
        extrapolate=False
    )

    print(f"Processed shape: {processed.shape}")
    print(f"Original sum: {np.sum(spectrum)} | Processed sum: {np.sum(processed)}")

    plt.figure(figsize=(15, 5))
    plt.plot(np.linspace(0, 20, 4096), np.sqrt(processed), label='Processed')
    plt.plot(np.linspace(0, 20, 4096), -np.sqrt(spectrum), label='Original')
    plt.xticks(np.arange(0, 21, 1))
    plt.grid()
    plt.xlabel("Energy (keV)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.savefig("temp.png")
