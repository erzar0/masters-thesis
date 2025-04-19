from .ArtifficialTrainDataGenerator import ArtifficialTrainDataGenerator
from scipy.optimize import curve_fit
from skimage.transform import resize
import numpy as np
import inspect
import warnings


class DataPreprocessor:
    """
    Provides static methods for preprocessing spectral data.
    """
    @staticmethod
    def processed_spectra(spectra, peaks, energies, min_energy=0, max_energy=20,
                          energy_margin=1, target_length=ArtifficialTrainDataGenerator.CHANNELS_COUNT,
                          polyfit_deg=1, extrapolate=True, extrapolation_kwargs=None):
        """
        Process spectra of shape (X, Y, C), calibrating, selecting energy range,
        resizing, optionally extrapolating, masking edges, and normalizing.

        Parameters:
            spectra (np.ndarray): Input spectra of shape (X, Y, C).
            peaks (np.ndarray): Channel positions of known energy peaks.
            energies (np.ndarray): Corresponding energy values for known peaks.
            min_energy (float): Minimum energy of the target range.
            max_energy (float): Maximum energy of the target range.
            energy_margin (float): Margin at spectra edges to mask with zeros.
            target_length (int): The desired number of channels (C dimension) in the output.
            polyfit_deg (int): Degree of polynomial for channel-to-energy calibration.
            extrapolate (bool): Whether to extrapolate padded regions.
            extrapolation_kwargs (dict, optional): Keyword arguments for _extrapolate_spectra.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - new_energy_range (np.ndarray): Target energy range of shape (target_length,).
                - processed_spectra (np.ndarray): Processed spectra of shape (X, Y, target_length).

        Raises:
            ValueError: If calibration results in insufficient points within the target energy range.
                        (Changed from returning zeros to raising error for clearer failure)
        """
        X, Y, C = spectra.shape
        peaks = np.array(peaks)
        energies = np.array(energies)
        
        if peaks.size < polyfit_deg + 1:
             raise ValueError(f"Cannot fit polynomial of degree {polyfit_deg} with only {peaks.size} points.")

        try:
            correction_coefficients = np.polyfit(peaks, energies, polyfit_deg)
            evaluate_polynomial = np.poly1d(correction_coefficients)
            original_energy_range = evaluate_polynomial(np.arange(C))
        except (np.linalg.LinAlgError, ValueError) as e:
            raise ValueError(f"Energy calibration failed using polyfit(deg={polyfit_deg}): {e}")


        mask = (original_energy_range >= min_energy) & (original_energy_range <= max_energy)
        masked_energy_range = original_energy_range[mask]
        masked_spectra = spectra[:, :, mask]

        if masked_energy_range.size < 2:
             raise ValueError(
                 f"Less than 2 data points found within the energy range "
                 f"[{min_energy}, {max_energy}] after calibration. Cannot proceed."
             )

        step_size = (masked_energy_range[1] - masked_energy_range[0]) # Safe now due to size check
        left_pad_width = max(0, int(np.floor((masked_energy_range[0] - min_energy) / step_size)))
        right_pad_width = max(0, int(np.ceil((max_energy - masked_energy_range[-1]) / step_size)))

        target_energy_range = np.linspace(min_energy, max_energy, target_length)
        padded_spectra = np.pad(masked_spectra, ((0, 0), (0, 0), (left_pad_width, right_pad_width)),
                                mode='constant', constant_values=0)

        resized_spectra = resize(padded_spectra,
                                 (X, Y, target_length),
                                 order=1,
                                 preserve_range=True,
                                 anti_aliasing=True if target_length < padded_spectra.shape[-1] else False)

        if extrapolate:
            if extrapolation_kwargs is None:
                extrapolation_kwargs = {}
            processed_spectra = DataPreprocessor._extrapolate_spectra(resized_spectra, target_energy_range, **extrapolation_kwargs)
        else:
            processed_spectra = resized_spectra

        edge_mask = np.ones_like(target_energy_range, dtype=processed_spectra.dtype)
        edge_mask[target_energy_range < (min_energy + energy_margin)] = 0
        edge_mask[target_energy_range > (max_energy - energy_margin)] = 0
        processed_spectra *= edge_mask

        min_vals = np.min(processed_spectra, axis=-1, keepdims=True)
        max_vals = np.max(processed_spectra, axis=-1, keepdims=True)
        denominator = max_vals - min_vals + 1e-10
        processed_spectra = np.divide(processed_spectra - min_vals, denominator,
                                     out=np.zeros_like(processed_spectra), where=denominator > 1e-9)


        return target_energy_range, processed_spectra

    @staticmethod
    def processed_spectrum(spectrum, peaks, energies, min_energy=0, max_energy=20,
                           energy_margin=1, target_length=ArtifficialTrainDataGenerator.CHANNELS_COUNT,
                           polyfit_deg=1, extrapolate=True, extrapolation_kwargs=None):
        """
        Process a single 1D spectrum using the processed_spectra method.

        Parameters:
            spectrum (np.ndarray): Input spectrum of shape (C,).
            peaks (np.ndarray): Channel positions of known energy peaks.
            energies (np.ndarray): Corresponding energy values for known peaks.
            min_energy (float): Minimum energy of the target range.
            max_energy (float): Maximum energy of the target range.
            energy_margin (float): Margin at spectra edges to mask with zeros.
                                   (Passed to processed_spectra).
            target_length (int): The desired number of channels in the output.
            polyfit_deg (int): Degree of polynomial for channel-to-energy calibration.
            extrapolate (bool): Whether to extrapolate padded regions.
            extrapolation_kwargs (dict, optional): Keyword arguments for _extrapolate_spectra.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - energy_range (np.ndarray): Target energy range of shape (target_length,).
                - processed_spectrum (np.ndarray): Processed spectrum of shape (target_length,).

         Raises:
            ValueError: If processing via processed_spectra fails.
        """
        if spectrum.ndim != 1:
            raise ValueError(f"Input spectrum must be 1D, but got shape {spectrum.shape}")

        reshaped_spectrum = spectrum.reshape(1, 1, -1)

        energy_range, processed_spectra_3d = DataPreprocessor.processed_spectra(
            reshaped_spectrum, peaks, energies, min_energy, max_energy,
            energy_margin, target_length, polyfit_deg, extrapolate, extrapolation_kwargs
        )

        processed_spectrum_1d = processed_spectra_3d.reshape(-1)

        return energy_range, processed_spectrum_1d


    @staticmethod
    def _extrapolate_spectra(spectra, energy_range, extrapolation_function='exponential',
                            channels_for_extrapolation=32, function_params=None,
                            curve_fit_kwargs=None):
        """
        Extrapolates spectra data using a specified function at ends where data is zero
        (typically resulting from padding before resize).

        Note: This operates pixel-by-pixel and can be slow for large X, Y dimensions.
        Note: Fits are performed on data *after* resizing.

        Args:
            spectra (np.ndarray): A 3D numpy array (X, Y, C) - data *after* padding and resizing.
            energy_range (np.ndarray): 1D array (C,) of corresponding energy values (target range).
            extrapolation_function (str or callable): Function ('exponential', 'quadratic') or callable(x, *params).
            channels_for_extrapolation (int): Max number of channels *from the edge of non-zero data*
                                              to use for fitting the extrapolation function.
            function_params (dict): Optional initial params `p0` for `curve_fit`, e.g., {'start': [..], 'end': [..]}.
            curve_fit_kwargs (dict): Optional kwargs for `scipy.optimize.curve_fit` (e.g., bounds, maxfev). Defaults to {'maxfev': 10000}.

        Returns:
            np.ndarray: The spectra data with ends potentially extrapolated. Modifies data in-place.
        """
        X, Y, C = spectra.shape
        extrapolated_spectra = np.copy(spectra)

        def exponential(x, a, b):
            return a * np.exp(b * x)

        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c

        function_map = {
            'exponential': exponential,
            'quadratic': quadratic
        }

        func = None
        num_params = 0
        if isinstance(extrapolation_function, str):
            if extrapolation_function not in function_map:
                raise ValueError(f"Invalid extrapolation function string: {extrapolation_function}. "
                                 f"Must be one of {list(function_map.keys())} or a callable.")
            func = function_map[extrapolation_function]
        elif callable(extrapolation_function):
             func = extrapolation_function
        else:
             raise TypeError("extrapolation_function must be a string or a callable.")

        try:
            sig = inspect.signature(func)
            num_params = len(sig.parameters) - 1
            if num_params <= 0:
                 raise ValueError("Extrapolation function must accept 'x' and at least one parameter.")
        except (TypeError, ValueError) as e:
             raise ValueError(f"Could not inspect extrapolation function '{func.__name__}': {e}")

        if curve_fit_kwargs is None:
            curve_fit_kwargs = {'maxfev': 10000}
        else:
             curve_fit_kwargs.setdefault('maxfev', 10000)

        warnings.warn("Performing pixel-wise extrapolation, which can be slow for large images.", RuntimeWarning)
        for i in range(X):
            for j in range(Y):
                current_spectrum = extrapolated_spectra[i, j, :]

                if np.all(current_spectrum == 0):
                    continue

                non_zero_indices = np.where(current_spectrum != 0)[0]
                if non_zero_indices.size == 0:
                     continue

                start_index = non_zero_indices[0]
                end_index = non_zero_indices[-1]

                if start_index > 0:
                    fit_start = start_index
                    fit_end = min(start_index + channels_for_extrapolation, end_index + 1)
                    x_data = energy_range[fit_start:fit_end]
                    y_data = current_spectrum[fit_start:fit_end]

                    if len(x_data) > num_params:
                        p0 = None
                        if function_params and 'start' in function_params:
                            p0 = function_params['start']
                        elif func == exponential: p0 = [np.mean(y_data) if np.mean(y_data)>0 else 1e-3 , -0.1]
                        elif func == quadratic: p0 = [1, 1, np.mean(y_data)]
                        elif not p0:
                             warnings.warn(f"No function_params['start'] provided for custom extrapolation function for spectrum ({i}, {j}). Fitting may fail.", RuntimeWarning)

                        try:
                            popt, _ = curve_fit(func, x_data, y_data, p0=p0, **curve_fit_kwargs)
                            extrapolated_values = func(energy_range[:start_index + 1], *popt)
                            extrapolated_values[extrapolated_values < 0] = 0
                            extrapolated_spectra[i, j, :start_index + 1] = extrapolated_values

                        except RuntimeError:
                            warnings.warn(f"curve_fit failed to converge for start extrapolation of spectrum ({i}, {j}). Skipping start extrapolation.", RuntimeWarning)
                        except ValueError as e:
                             warnings.warn(f"curve_fit error during start extrapolation for spectrum ({i}, {j}): {e}. Skipping start extrapolation.", RuntimeWarning)

                    else:
                         warnings.warn(f"Insufficient data points ({len(x_data)}) for start extrapolation (needs > {num_params}) for spectrum ({i}, {j}). Skipping start extrapolation.", RuntimeWarning)

                if end_index < C - 1:
                    fit_end = end_index + 1
                    fit_start = max(0, end_index - channels_for_extrapolation + 1)
                    x_data = energy_range[fit_start:fit_end]
                    y_data = current_spectrum[fit_start:fit_end]

                    if len(x_data) > num_params:
                        p0 = None 
                        if function_params and 'end' in function_params:
                            p0 = function_params['end']
                        elif func == exponential: p0 = [y_data[-1] if y_data[-1]>0 else 1e-3 , -0.1]
                        elif func == quadratic: p0 = [1, 1, y_data[-1]]
                        elif not p0:
                            warnings.warn(f"No function_params['end'] provided for custom extrapolation function for spectrum ({i}, {j}). Fitting may fail.", RuntimeWarning)

                        try:
                            popt, _ = curve_fit(func, x_data, y_data, p0=p0, **curve_fit_kwargs)
                            extrapolated_values = func(energy_range[end_index:], *popt)
                            extrapolated_values[extrapolated_values < 0] = 0
                            extrapolated_spectra[i, j, end_index:] = extrapolated_values

                        except RuntimeError:
                            warnings.warn(f"curve_fit failed to converge for end extrapolation of spectrum ({i}, {j}). Skipping end extrapolation.", RuntimeWarning)
                        except ValueError as e:
                            warnings.warn(f"curve_fit error during end extrapolation for spectrum ({i}, {j}): {e}. Skipping end extrapolation.", RuntimeWarning)

                    else:
                        warnings.warn(f"Insufficient data points ({len(x_data)}) for end extrapolation (needs > {num_params}) for spectrum ({i}, {j}). Skipping end extrapolation.", RuntimeWarning)

        return extrapolated_spectra
    
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

    from random import randint

    @staticmethod
    def find_heuristic_subtraction(
        a: np.ndarray,
        b: np.ndarray,
        tf_start: float = 0.05,
        tf_stop: float = 0.50,
        tf_step: float = 0.01,
        verbose: bool = False,
        **kwargs # Pass other args like tol, max_iter to bisect_subtract_residual
    ) -> tuple[np.ndarray | None, float | None]:
        """
        Finds a "good" background subtraction result using a heuristic approach.

        It iterates through a range of `target_fraction` values for the
        `bisect_subtract_residual` function, calculates the average of all
        successfully generated results, and returns the individual result that
        is closest (in terms of Mean Squared Error) to this average.

        Parameters:
            a (np.ndarray): Original spectrum (should be non-negative).
            b (np.ndarray): Spectrum to subtract (residual, should be non-negative).
            tf_start (float): Starting value for target_fraction range.
            tf_stop (float): Ending value for target_fraction range (exclusive).
            tf_step (float): Step size for target_fraction range.
            verbose (bool): If True, prints warnings when a target_fraction fails.
            **kwargs: Additional keyword arguments to pass directly to
                    `bisect_subtract_residual` (e.g., tol, max_iter, epsilon).

        Returns:
            tuple[np.ndarray | None, float | None]:
                - The resulting spectrum deemed closest to the average of results
                across the target_fraction range. Returns None if no spectra
                could be generated.
                - The target_fraction value that produced the returned spectrum.
                Returns None if no spectra could be generated.

        Raises:
            ValueError: If spectra shapes mismatch.
            RuntimeError: If no spectra could be successfully generated across the
                        entire target_fraction range.
        """
        if a.shape != b.shape:
            raise ValueError("Spectra `a` and `b` must have the same shape.")

        results_list = []
        tfs_used = []
        target_fractions = np.arange(tf_start, tf_stop, tf_step)

        if len(target_fractions) == 0:
            warnings.warn("Target fraction range is empty, returning None.", RuntimeWarning)
            return None, None

        # --- Generate results for each target_fraction ---
        for tf in target_fractions:
            try:
                res = DataPreprocessor.bisect_subtract_residual(a, b, target_fraction=tf, **kwargs)
                results_list.append(res)
                tfs_used.append(tf)
            except (RuntimeError, ValueError) as e:
                if verbose:
                    warnings.warn(
                        f"Skipping target_fraction={tf:.3f} due to error: {e}",
                        RuntimeWarning
                    )
                continue # Skip this target_fraction

        if not results_list:
            raise RuntimeError(
                f"Could not generate any valid subtracted spectra for the "
                f"target_fraction range [{tf_start}, {tf_stop}). Check inputs "
                f"or bisect_subtract_residual parameters."
            )

        # --- Calculate average and find closest spectrum ---
        # Convert list of 1D arrays to a 2D array for averaging
        results_array = np.array(results_list)
        average_spectrum = np.mean(results_array, axis=0)

        min_mse = np.inf
        closest_spectrum = None
        best_tf = None

        for i, spectrum in enumerate(results_list):
            mse = np.mean((spectrum - average_spectrum)**2)
            if mse < min_mse:
                min_mse = mse
                closest_spectrum = spectrum
                best_tf = tfs_used[i] # Get the tf corresponding to this spectrum

        return closest_spectrum, best_tf
