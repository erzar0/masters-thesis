from .Elements import Elements
from tqdm import tqdm 
from typing import Dict
from scipy.stats import beta
import numpy as np
import random 

class SemiArtifficialTrainDataGenerator:
    TARGET_VECTOR_LENGTH = -1
    CACHED_SPECTRUM_SAMPLES = {}
    SPECTRA_NAMES = []
    CHANNELS_COUNT = 4096

    @staticmethod
    def _get_random_percentages(n):
        """Generate a list of random percentages that sum to 1."""
        percentages = np.random.uniform(0, 1, n)
        return percentages / np.sum(percentages)

    @staticmethod
    def generate_artificial_data(spectra_samples: Dict[str, np.array], samples_count: int =10000):
        """Generate artificial data based on the provided spectra samples."""
        unique_elements = set()
        for name, spectrum in spectra_samples.items():
            for element_name in name.split("_"):
                unique_elements.add(element_name)
            SemiArtifficialTrainDataGenerator.CACHED_SPECTRUM_SAMPLES[name] = spectrum
        
        print(unique_elements) 
        SemiArtifficialTrainDataGenerator.TARGET_VECTOR_LENGTH = len(unique_elements)
        SemiArtifficialTrainDataGenerator.SPECTRA_NAMES = list(spectra_samples.keys())

        spectra_counts = beta.rvs(a=4, b=20, size=samples_count)
        different_spectra_count = len(SemiArtifficialTrainDataGenerator.SPECTRA_NAMES)
        spectra_counts = np.clip((spectra_counts * different_spectra_count).astype(int) + 1, 1, different_spectra_count).tolist()

        X = []
        y = []

        for selected_spectra_count in tqdm(spectra_counts):
            selected_spectra_names = random.sample(SemiArtifficialTrainDataGenerator.SPECTRA_NAMES, selected_spectra_count)
            spectrum_data = np.zeros(SemiArtifficialTrainDataGenerator.CHANNELS_COUNT)
            target_vector = np.zeros(SemiArtifficialTrainDataGenerator.TARGET_VECTOR_LENGTH)
            random_percentages = SemiArtifficialTrainDataGenerator._get_random_percentages(selected_spectra_count)

            for i, name in enumerate(selected_spectra_names):
                spectrum_data += SemiArtifficialTrainDataGenerator.CACHED_SPECTRUM_SAMPLES[name] * random_percentages[i]
                for element_name in name.split("_"):
                    target_vector[Elements.SYMBOL2NUM[element_name]] = random_percentages[i]

            spectrum_data /= np.max(spectrum_data)
            
            X.append(spectrum_data)
            y.append(target_vector)
        
        return np.array(X), np.array(y)

        








