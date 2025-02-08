import numpy as np
from .TrainDataGenerator import TrainDataGenerator
from skimage.transform import resize


import numpy as np

class DataPreprocessor:
    @staticmethod
    def processed_spectra(spectra, peaks, energies, min_energy=0, max_energy=20, energy_margin=1, target_length=TrainDataGenerator.CHANNELS_COUNT):
        """
        Process a spectra of shape (X, Y, C), transforming the last dimension (C).
        
        Parameters:
            spectra (np.ndarray): Input spectra of shape (X, Y, C)
            peaks (np.ndarray): Channel positions of known energy peaks
            energies (np.ndarray): Corresponding energy values for known peaks
            min_energy (float): Minimum energy range to consider
            max_energy (float): Maximum energy range to consider
            energy_margin (float): Margin at the spectra edges to reduce noise

        Returns:
            new_energy_range (np.ndarray): New energy range of shape (C,)
            processed_spectra (np.ndarray): Processed spectra of shape (X, Y, C)
        """
        X, Y, C = spectra.shape

        # Compute channel-to-energy mapping
        correction_coefficients = np.polyfit(peaks, energies, 1)
        evaluate_polynomial = np.poly1d(correction_coefficients)
        energy_range = evaluate_polynomial(np.arange(C))

        # Create a mask for valid energy values
        mask = (energy_range >= min_energy) & (energy_range <= max_energy)
        
        # Apply mask across all (X, Y)
        energy_range = energy_range[mask]
        spectra = spectra[:, :, mask]

        # Compute margin index based on energy range
        energy_margin_idx = int((energy_margin / (max_energy - min_energy)) * spectra.shape[-1])
        
        # Apply edge masking (reduce noise)
        spectra[:, :, :energy_margin_idx] = 0
        spectra[:, :, -energy_margin_idx:] = 0

        # Padding along last axis (C) to match [min_energy, max_energy] range 
        step_size = (energy_range[1] - energy_range[0])
        left_pad_width = max(0, int((energy_range[0] - min_energy) / step_size))
        right_pad_width = max(0, int((max_energy - energy_range[-1]) / step_size))

        energy_range = np.pad(energy_range, (left_pad_width, right_pad_width), mode='linear_ramp')
        spectra = np.pad(spectra, ((0, 0), (0, 0), (left_pad_width, right_pad_width)), mode='constant', constant_values=0)

        # Resize to target length
        spectra = resize(spectra, (spectra.shape[0], spectra.shape[1], target_length), anti_aliasing=True if target_length < spectra.shape[-1] else False)
        energy_range = np.linspace(energy_range[0], energy_range[-1], target_length)

        # Normalize spectra per (X, Y) individually
        min_vals = np.min(spectra, axis=-1, keepdims=True)
        max_vals = np.max(spectra, axis=-1, keepdims=True)
        spectra = (spectra - min_vals) / (max_vals + min_vals + 1e-10)

        return  energy_range, spectra

    @staticmethod
    def processed_spectrum(spectrum, peaks, energies, min_energy=0, max_energy=20, energy_margin=1):
        energy_range, spectra = DataPreprocessor.processed_spectra(spectrum.reshape(1, 1, spectrum.shape[-1]), peaks, energies, min_energy, max_energy, energy_margin)
        return energy_range, spectra.reshape(-1)