from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from skimage.transform import resize
from tqdm import tqdm


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
        target_length: int = 4096,
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
        discrete_transformation: bool = False,
        **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        if spectrum.ndim != 1:
            raise ValueError("Input spectrum must be 1D.")
        reshaped = spectrum.reshape(1, 1, -1)
        energy_range, processed = DataPreprocessor.processed_spectra(reshaped, peaks, energies, discrete_transformation=discrete_transformation, **kwargs)
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
    
    @staticmethod
    def bisect_subtract_residual(
        a: np.ndarray,
        b: np.ndarray,
        target_fraction: float = 0.1,
        tol: float = 0.01,
        max_iter: int = 100, # Bisection converges fast, 100 is usually plenty
        max_init_iter: int = 20, # Max iterations for finding initial alpha_high
        initial_alpha_high_factor: float = 0.5, # Original heuristic factor
        epsilon: float = 1e-9 # Small value to prevent division by zero
    ) -> np.ndarray:
        """
        Subtracts spectrum `b` from `a` using bisection until `target_fraction`
        of resulting values are negative. Includes robust initial bound finding
        and safe normalization.

        Parameters:
            a (np.ndarray): Original spectrum (should be non-negative).
            b (np.ndarray): Spectrum to subtract (residual, should be non-negative).
            target_fraction (float): Target fraction of negative values in the
                                    intermediate result (a - alpha*b). Must be > 0.
            tol (float): Tolerance for stopping bisection (difference between
                        actual and target negative fraction).
            max_iter (int): Maximum number of bisection iterations.
            max_init_iter (int): Maximum iterations for finding initial upper bound.
            initial_alpha_high_factor (float): Initial guess factor for alpha_high
                                            relative to peak ratios.
            epsilon (float): Small number to prevent division by zero.

        Returns:
            np.ndarray: Adjusted spectrum (a - alpha*b) with negative values clipped
                        to zero and normalized to max value of 1. Returns array of
                        zeros if the result is all non-positive after subtraction
                        and clipping.

        Raises:
            ValueError: If spectra shapes mismatch, or if target_fraction <= 0.
            RuntimeError: If a suitable initial upper bound alpha_high cannot be found.
        """
        if a.shape != b.shape:
            raise ValueError("Spectra `a` and `b` must have the same shape.")
        if not (0 < target_fraction < 1):
            raise ValueError("target_fraction must be between 0 and 1 (exclusive).")

        # --- Robust Initial Bounds ---
        alpha_low = 0.0

        # Calculate initial guess for alpha_high
        max_a = np.max(a)
        max_b = np.max(b)

        if max_b <= epsilon: # Already checked above, but belt-and-suspenders
            alpha_high = 1.0 # Arbitrary guess, b is zero anyway
        else:
            # Heuristic: scale so peaks roughly match, then scale by factor
            alpha_high = (max_a / max_b) * initial_alpha_high_factor

        # Ensure alpha_high is at least slightly positive
        alpha_high = max(alpha_high, epsilon)

        # Check if initial alpha_high produces enough negative values.
        # If not, increase it until it does or max_init_iter is reached.
        iter_init = 0
        while np.mean((a - alpha_high * b) < 0) <= target_fraction and iter_init < max_init_iter :
            # Increase alpha_high exponentially
            alpha_high *= 2.0
            iter_init += 1

        # Check if we found a valid upper bound
        if iter_init == max_init_iter and np.mean((a - alpha_high * b) < 0) <= target_fraction:
            # This could happen if e.g. target_fraction is very high and 'a' always dominates 'b'
            raise RuntimeError(
                f"Could not find an initial alpha_high (tried up to {alpha_high:.2e}) "
                f"that produces a negative fraction > {target_fraction:.3f}. "
                f"Check inputs or target_fraction."
            )


        # --- Bisection Search ---
        final_alpha = alpha_low # Default if loop doesn't run
        for i in range(max_iter):
            alpha = (alpha_low + alpha_high) / 2
            result_intermediate = a - alpha * b

            negative_fraction = np.mean(result_intermediate < 0)

            # Check for convergence
            if abs(negative_fraction - target_fraction) < tol:
                final_alpha = alpha
                break

            # Narrow the interval
            if negative_fraction > target_fraction:
                # Too many negatives, alpha is too high
                alpha_high = alpha
            else:
                # Too few negatives, alpha is too low
                alpha_low = alpha
            final_alpha = alpha # Store last value in case max_iter is reached

        else: # nobreak - loop finished without converging within tolerance
            warnings.warn(
                f"Bisection did not converge within {tol=:.3g} after {max_iter} iterations. "
                f"Final negative fraction = {negative_fraction:.3f} (target = {target_fraction:.3f}). "
                f"Using alpha = {final_alpha:.3e}.",
                RuntimeWarning
            )

        # --- Post-Processing ---
        # Calculate final result with the determined alpha
        result = a - final_alpha * b

        # Clip negative values to zero
        result[result < 0] = 0

        # Normalize safely
        max_val = np.max(result)
        if max_val > epsilon: # Use epsilon here for floating point comparison
            result = result / max_val

        return result



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
