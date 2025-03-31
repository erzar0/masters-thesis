from .TrainDataGenerator import TrainDataGenerator
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
                          energy_margin=1, target_length=TrainDataGenerator.CHANNELS_COUNT,
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

        # --- 1. Channel-to-Energy Calibration ---
        try:
            correction_coefficients = np.polyfit(peaks, energies, polyfit_deg)
            evaluate_polynomial = np.poly1d(correction_coefficients)
            original_energy_range = evaluate_polynomial(np.arange(C))
        except (np.linalg.LinAlgError, ValueError) as e:
            raise ValueError(f"Energy calibration failed using polyfit(deg={polyfit_deg}): {e}")


        # --- 2. Energy Range Masking ---
        mask = (original_energy_range >= min_energy) & (original_energy_range <= max_energy)
        masked_energy_range = original_energy_range[mask]
        masked_spectra = spectra[:, :, mask]

        # Check if enough data remains after masking
        if masked_energy_range.size < 2:
             # Changed behavior: Raise error instead of returning zeros for clearer failure indication
             raise ValueError(
                 f"Less than 2 data points found within the energy range "
                 f"[{min_energy}, {max_energy}] after calibration. Cannot proceed."
             )

        # --- 3. Padding Calculation ---
        # Calculate approximate step size from the *masked* data for padding estimation
        # Note: Steps might not be perfectly uniform if polyfit_deg > 1
        step_size = (masked_energy_range[1] - masked_energy_range[0]) # Safe now due to size check
        # Use np.floor/ceil for potentially more robust padding width calculation
        left_pad_width = max(0, int(np.floor((masked_energy_range[0] - min_energy) / step_size)))
        right_pad_width = max(0, int(np.ceil((max_energy - masked_energy_range[-1]) / step_size)))

        # --- 4. Define Target Energy Range and Pad Spectra ---
        # This is the final, uniform energy range we are mapping onto
        target_energy_range = np.linspace(min_energy, max_energy, target_length)
        # Pad the *masked* spectra. The padding aims to place the existing data
        # roughly correctly within the [min_energy, max_energy] span before resizing.
        padded_spectra = np.pad(masked_spectra, ((0, 0), (0, 0), (left_pad_width, right_pad_width)),
                                mode='constant', constant_values=0)

        # --- 5. Resize to Target Length ---
        # Resize the padded spectra to match the target_length.
        # preserve_range=True prevents resize from scaling values to [0,1] itself.
        # order=1 specifies bilinear/trilinear interpolation.
        # anti_aliasing is important when downsampling.
        # Note: This step interpolates the data, including the padded zeros near the edges.
        resized_spectra = resize(padded_spectra,
                                 (X, Y, target_length),
                                 order=1,
                                 preserve_range=True,
                                 anti_aliasing=True if target_length < padded_spectra.shape[-1] else False)

        # --- 6. Extrapolation (Optional) ---
        # Extrapolate the regions that were originally padded zeros, now potentially
        # modified by the resize interpolation. This happens *after* resizing.
        if extrapolate:
            if extrapolation_kwargs is None:
                extrapolation_kwargs = {} # Use defaults in _extrapolate_spectra
            # Pass the resized spectra and the final target energy range
            processed_spectra = DataPreprocessor._extrapolate_spectra(resized_spectra, target_energy_range, **extrapolation_kwargs)
        else:
            processed_spectra = resized_spectra # Keep the resized data as is

        # --- 7. Edge Masking (Noise Reduction) ---
        # Create mask based on the final target energy range
        edge_mask = np.ones_like(target_energy_range, dtype=processed_spectra.dtype) # Match dtype
        edge_mask[target_energy_range < (min_energy + energy_margin)] = 0
        edge_mask[target_energy_range > (max_energy - energy_margin)] = 0
        # Apply mask (broadcasts across X, Y)
        processed_spectra *= edge_mask

        # --- 8. Normalization (Per Spectrum) ---
        # Normalize each spectrum individually to [0, 1] range.
        min_vals = np.min(processed_spectra, axis=-1, keepdims=True)
        max_vals = np.max(processed_spectra, axis=-1, keepdims=True)
        # Using standard min-max scaling denominator
        denominator = max_vals - min_vals + 1e-10 # Add epsilon for stability
        # Handle case where max == min (flat spectrum) -> results in zeros
        processed_spectra = np.divide(processed_spectra - min_vals, denominator,
                                     out=np.zeros_like(processed_spectra), where=denominator > 1e-9)


        return target_energy_range, processed_spectra

    @staticmethod
    def processed_spectrum(spectrum, peaks, energies, min_energy=0, max_energy=20,
                           energy_margin=1, target_length=TrainDataGenerator.CHANNELS_COUNT,
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

        # Reshape to (1, 1, C) for processed_spectra
        reshaped_spectrum = spectrum.reshape(1, 1, -1)

        # Call the main processing function
        energy_range, processed_spectra_3d = DataPreprocessor.processed_spectra(
            reshaped_spectrum, peaks, energies, min_energy, max_energy,
            energy_margin, target_length, polyfit_deg, extrapolate, extrapolation_kwargs
        )

        # Reshape result back to 1D
        processed_spectrum_1d = processed_spectra_3d.reshape(-1)

        return energy_range, processed_spectrum_1d


    @staticmethod
    def _extrapolate_spectra(spectra, energy_range, extrapolation_function='exponential',
                            channels_for_extrapolation=128, function_params=None,
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
        extrapolated_spectra = np.copy(spectra) # Work on a copy

        # --- Define extrapolation model functions ---
        def exponential(x, a, b):
            # Add safeguards for large exponents if needed
            # with np.errstate(over='ignore', under='ignore'):
            return a * np.exp(b * x)

        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c

        function_map = {
            'exponential': exponential,
            'quadratic': quadratic
        }

        # --- Resolve extrapolation function ---
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
            # Inspect the chosen function to find number of parameters (excluding 'x')
            sig = inspect.signature(func)
            num_params = len(sig.parameters) - 1
            if num_params <= 0:
                 raise ValueError("Extrapolation function must accept 'x' and at least one parameter.")
        except (TypeError, ValueError) as e:
             raise ValueError(f"Could not inspect extrapolation function '{func.__name__}': {e}")


        # --- Default curve_fit settings ---
        if curve_fit_kwargs is None:
            curve_fit_kwargs = {'maxfev': 10000}
        else:
            # Ensure maxfev has a default if not provided
             curve_fit_kwargs.setdefault('maxfev', 10000)


        # --- Pixel-wise Extrapolation (Performance Bottleneck) ---
        warnings.warn("Performing pixel-wise extrapolation, which can be slow for large images.", RuntimeWarning)
        for i in range(X):
            for j in range(Y):
                current_spectrum = extrapolated_spectra[i, j, :]

                # Skip if spectrum is all zero
                if np.all(current_spectrum == 0):
                    continue

                # Find indices of non-zero elements
                non_zero_indices = np.where(current_spectrum != 0)[0]
                if non_zero_indices.size == 0:
                     # Should have been caught by the np.all check, but as a safeguard
                     continue

                start_index = non_zero_indices[0]
                end_index = non_zero_indices[-1]

                # --- Extrapolate Start ---
                if start_index > 0: # Need to extrapolate if non-zero data doesn't start at index 0
                    # Select data *from* the first non-zero point onwards for fitting
                    fit_start = start_index
                    fit_end = min(start_index + channels_for_extrapolation, end_index + 1)
                    x_data = energy_range[fit_start:fit_end]
                    y_data = current_spectrum[fit_start:fit_end]

                    if len(x_data) > num_params: # Check if enough points exist
                        p0 = None # Determine p0
                        if function_params and 'start' in function_params:
                            p0 = function_params['start']
                        elif func == exponential: p0 = [np.mean(y_data) if np.mean(y_data)>0 else 1e-3 , -0.1] # Adjusted default p0
                        elif func == quadratic: p0 = [1, 1, np.mean(y_data)] # Adjusted default p0
                        elif not p0:
                             warnings.warn(f"No function_params['start'] provided for custom extrapolation function for spectrum ({i}, {j}). Fitting may fail.", RuntimeWarning)

                        try:
                            # Curve fit
                            popt, _ = curve_fit(func, x_data, y_data, p0=p0, **curve_fit_kwargs)
                            # Apply extrapolation from index 0 up to start_index (inclusive)
                            # Note: Overwrites original value at start_index for continuity
                            extrapolated_values = func(energy_range[:start_index + 1], *popt)
                            # Prevent negative values if physically meaningless
                            extrapolated_values[extrapolated_values < 0] = 0
                            extrapolated_spectra[i, j, :start_index + 1] = extrapolated_values

                        except RuntimeError:
                            warnings.warn(f"curve_fit failed to converge for start extrapolation of spectrum ({i}, {j}). Skipping start extrapolation.", RuntimeWarning)
                        except ValueError as e:
                             warnings.warn(f"curve_fit error during start extrapolation for spectrum ({i}, {j}): {e}. Skipping start extrapolation.", RuntimeWarning)

                    else:
                         warnings.warn(f"Insufficient data points ({len(x_data)}) for start extrapolation (needs > {num_params}) for spectrum ({i}, {j}). Skipping start extrapolation.", RuntimeWarning)

                # --- Extrapolate End ---
                if end_index < C - 1: # Need to extrapolate if non-zero data doesn't end at last index
                    # Select data *up to* the last non-zero point for fitting
                    fit_end = end_index + 1 # Corrected slicing: include end_index
                    fit_start = max(0, end_index - channels_for_extrapolation + 1)
                    x_data = energy_range[fit_start:fit_end]
                    y_data = current_spectrum[fit_start:fit_end]

                    if len(x_data) > num_params: # Check if enough points exist
                        p0 = None # Determine p0
                        if function_params and 'end' in function_params:
                            p0 = function_params['end']
                        elif func == exponential: p0 = [y_data[-1] if y_data[-1]>0 else 1e-3 , -0.1] # Adjusted default p0
                        elif func == quadratic: p0 = [1, 1, y_data[-1]] # Adjusted default p0
                        elif not p0:
                            warnings.warn(f"No function_params['end'] provided for custom extrapolation function for spectrum ({i}, {j}). Fitting may fail.", RuntimeWarning)

                        try:
                            # Curve fit
                            popt, _ = curve_fit(func, x_data, y_data, p0=p0, **curve_fit_kwargs)
                            # Apply extrapolation from end_index up to the end of the array
                            # Note: Overwrites original value at end_index for continuity
                            extrapolated_values = func(energy_range[end_index:], *popt)
                            # Prevent negative values if physically meaningless
                            extrapolated_values[extrapolated_values < 0] = 0
                            extrapolated_spectra[i, j, end_index:] = extrapolated_values

                        except RuntimeError:
                            warnings.warn(f"curve_fit failed to converge for end extrapolation of spectrum ({i}, {j}). Skipping end extrapolation.", RuntimeWarning)
                        except ValueError as e:
                            warnings.warn(f"curve_fit error during end extrapolation for spectrum ({i}, {j}): {e}. Skipping end extrapolation.", RuntimeWarning)

                    else:
                        warnings.warn(f"Insufficient data points ({len(x_data)}) for end extrapolation (needs > {num_params}) for spectrum ({i}, {j}). Skipping end extrapolation.", RuntimeWarning)

        return extrapolated_spectra # Return the modified copy
