import numpy as np

from .FeatureEnhancer import FeatureEnhancer
from .TrainDataGenerator import TrainDataGenerator
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

class DataPreprocessor():
    @staticmethod
    def get_processed_spectrum(spectrum, peaks, energies, min_energy=0, max_energy=20, energy_margin=1, target_length=TrainDataGenerator.CHANNELS_COUNT):
        # Calculate channel to energy mapping
        correction_coefficients = np.polyfit(peaks, energies, 1)
        energy_range = np.poly1d(correction_coefficients)(np.arange(spectrum.shape[-1]))

        # Fit spectrum within the energy range [min_energy, max_energy]
        mask = (energy_range >= min_energy) & (energy_range <= max_energy)
        energy_range_mapped = energy_range[mask]
        spectrum_mapped = spectrum[mask]

        # Mask edges of spectrum by energy_margin to reduce noise
        energy_margin_idx = int((energy_margin / (max_energy - min_energy)) * (spectrum_mapped.shape[-1]))
        spectrum_mapped[: int(energy_margin_idx)] = spectrum_mapped[energy_margin_idx]
        spectrum_mapped[-int(energy_margin_idx): ] = spectrum_mapped[-energy_margin_idx]

        # Apply gaussian filter over spectrum
        # spectrum_mapped = gaussian_filter1d()

        # Interpolate to achieve target_length
        # interpolation_energy_range = np.linspace(energy_range_mapped[0], energy_range_mapped[-1], target_length)
        # spectrum_interpolated = interp1d(energy_range_mapped, spectrum_mapped, kind='linear', fill_value="extrapolate")(interpolation_energy_range)

        # Normalize spectrum
        # spectrum_interpolated = spectrum_interpolated / (np.max(spectrum_interpolated) + 10e-10)

        return energy_range_mapped, spectrum_mapped / (np.max(spectrum_mapped) + 10e-10)
        


    @staticmethod
    def process(e_xy, peaks, energies, min_energy=0, max_energy=20, energy_margin=1, target_length=TrainDataGenerator.CHANNELS_COUNT, enhance=True):
        result = np.zeros((*e_xy.shape[:-1], target_length))
        energy_range = None
        for i in tqdm(range(e_xy.shape[0]), desc="processing data row"):
            for j in range(e_xy.shape[1]):
                energy_range, result[i, j] = DataPreprocessor.get_processed_spectrum(e_xy[i, j], peaks, energies, min_energy, max_energy, energy_margin, target_length)

        if enhance:
            previous_shape = result.shape
            result = FeatureEnhancer.enhanced_features(result.reshape(-1, target_length))
            new_shape = result.shape
            result = result.reshape((*previous_shape[:-1], *new_shape[1:]))

        return energy_range, result