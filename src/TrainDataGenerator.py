import numpy as np
import random 

from .Elements import Elements
from functools import reduce
from tqdm import tqdm 

class TrainDataGenerator():
    CHANNELS_COUNT = 4096
    TARGET_VECTOR_LENGTH = len(Elements.LINES.keys())
    CACHED_ELEMENT_SAMPLES = {}
    MIN_ENERGY = 0
    MAX_ENERGY = 20

    @staticmethod
    def gaussian(x, mu, sigma):
        """Return Gaussian distribution values."""
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    @staticmethod
    def _get_random_percentages(n):
        """Generate a list of random percentages that sum to 1."""
        percentages = np.random.uniform(0, 1, n)
        return percentages / np.sum(percentages)

    @staticmethod
    def _sample_elements(elements_count, selected_elements=None):
        """Sample a given number of elements."""
        if selected_elements is None:
            return random.sample(list(Elements.LINES.keys()), elements_count)
        return [el if el in Elements.LINES.keys() else Elements.NUM2SYMBOL[el] for el in selected_elements]

    @staticmethod
    def _add_escape_peak(energy_range, element_line, element_sample, mu_err_global, mu_max_err, sigma_max_err, scale_sigma):
        """Add escape peak to the element sample if conditions are met."""
        mu_esc = element_line["mu"] - Elements.ESCAPE_ENERGY_DIFF + np.random.uniform(-mu_max_err, mu_max_err) + mu_err_global
        sigma = Elements.calculate_sigma(mu_esc) * scale_sigma + np.random.uniform(-sigma_max_err, sigma_max_err)
        intensity = element_line["intensity"] * Elements.ESCAPE_ENERGY_INTENSITY_RATIO
        gaussian = TrainDataGenerator.gaussian(energy_range, mu_esc, sigma)
        gaussian /= np.max(gaussian)
        gaussian *= intensity
        element_sample += gaussian
        return element_sample

    @staticmethod
    def _add_cu_peak(energy_range, element_line, element_sample, mu_err_global, mu_max_err, sigma_max_err, scale_sigma):
        """Add copper peak to the element sample if conditions are met."""
        mu_cu = Elements.CU_THRESHOLD_ENERGY + np.random.uniform(-mu_max_err, mu_max_err) + mu_err_global
        sigma = Elements.calculate_sigma(mu_cu) * scale_sigma + np.random.uniform(-sigma_max_err, sigma_max_err)
        intensity = element_line["intensity"] * Elements.ESCAPE_ENERGY_INTENSITY_RATIO
        gaussian = TrainDataGenerator.gaussian(energy_range, mu_cu, sigma)
        gaussian /= np.max(gaussian)
        gaussian *= intensity
        element_sample += gaussian
        return element_sample

    @staticmethod
    def _generate_element_sample(energy_range, element, mu_err_global, mu_max_err, sigma_max_err, scale_sigma, cache_element_samples):
        """Generate a sample for a single element."""
        if cache_element_samples and element in TrainDataGenerator.CACHED_ELEMENT_SAMPLES:
            return TrainDataGenerator.CACHED_ELEMENT_SAMPLES[element].copy()

        element_sample = np.zeros(TrainDataGenerator.CHANNELS_COUNT)
        for element_line in Elements.get_parsed_element_lines(element):
            mu = element_line["mu"] + np.random.uniform(-mu_max_err, mu_max_err) + mu_err_global
            sigma = element_line["sigma"] * scale_sigma + np.random.uniform(-sigma_max_err, sigma_max_err)
            intensity = element_line["intensity"]
            gaussian = TrainDataGenerator.gaussian(energy_range, mu, sigma)
            gaussian /= np.max(gaussian)
            gaussian *= intensity
            element_sample += gaussian

            if mu > Elements.ESC_THRESHOLD_ENERGY:
                element_sample = TrainDataGenerator._add_escape_peak(energy_range, element_line, element_sample, mu_err_global, mu_max_err, sigma_max_err, scale_sigma)
            if mu > Elements.CU_THRESHOLD_ENERGY:
                element_sample = TrainDataGenerator._add_cu_peak(energy_range, element_line, element_sample, mu_err_global, mu_max_err, sigma_max_err, scale_sigma)


        element_sample /= np.max(element_sample)

        if element not in TrainDataGenerator.CACHED_ELEMENT_SAMPLES:
            TrainDataGenerator.CACHED_ELEMENT_SAMPLES[element] = element_sample

        return element_sample

    @staticmethod
    def generate_sample(energy_range, selected_elements, element_percentages, mu_err_global, mu_max_err, sigma_max_err, scale_sigma=1, set_percentages=False, cache_element_samples=False):
        """Generate a single sample with specified elements and parameters."""
        sample = np.zeros(TrainDataGenerator.CHANNELS_COUNT)
        target = np.zeros(TrainDataGenerator.TARGET_VECTOR_LENGTH)

        for element_percentage, element in zip(element_percentages, selected_elements):
            target[Elements.SYMBOL2NUM[element]] = element_percentage if set_percentages else 1
            element_sample = TrainDataGenerator._generate_element_sample(energy_range, element, mu_err_global, mu_max_err, sigma_max_err, scale_sigma, cache_element_samples)
            element_sample *= element_percentage
            sample += element_sample

        sample /= np.max(sample)
        return sample, target

    @staticmethod
    def generate_many_samples(samples=10, elements_per_sample=3, mu_max_err=0.0, mu_max_err_global=0.0, sigma_max_err=0.0, elements=None, use_percentages=False, cache_element_samples=False):
        """Generate multiple samples with specified parameters."""
        X = [np.zeros(TrainDataGenerator.CHANNELS_COUNT) for _ in range(samples)]
        y = [np.zeros(TrainDataGenerator.TARGET_VECTOR_LENGTH) for _ in range(samples)]
        energy_range = np.linspace(TrainDataGenerator.MIN_ENERGY, TrainDataGenerator.MAX_ENERGY, TrainDataGenerator.CHANNELS_COUNT)

        for i in tqdm(range(samples), desc="generating artificial data samples"):
            selected_elements = TrainDataGenerator._sample_elements(elements_per_sample, elements)
            element_percentages = TrainDataGenerator._get_random_percentages(len(selected_elements))
            mu_err_global = np.random.uniform(-mu_max_err_global, mu_max_err_global)
            exponential = np.exp(-np.linspace(4, 5, TrainDataGenerator.CHANNELS_COUNT))

            sample, target = TrainDataGenerator.generate_sample(energy_range, selected_elements, element_percentages, mu_err_global, mu_max_err, sigma_max_err, scale_sigma=1, set_percentages=use_percentages, cache_element_samples=cache_element_samples)

            sample += exponential
            sample /= np.max(sample)
            X[i] = sample
            y[i] = target
        return X, y

    @staticmethod
    def generate_artificial_data(samples=10000, max_elements_per_sample=1, use_max=False, mu_max_err=0.0, mu_max_err_global=0.0, sigma_max_err=0.0, use_percentages=False, cache_element_samples=False):
        max_elements_per_sample += 1
        if use_max:
            artificial_data = [TrainDataGenerator.generate_many_samples(samples=samples, elements_per_sample=max_elements_per_sample, mu_max_err=0.0, mu_max_err_global=mu_max_err_global, sigma_max_err=sigma_max_err, elements=None, use_percentages=False, cache_element_samples=True)]

        else:
            artificial_data = [TrainDataGenerator.generate_many_samples(samples=samples // (max_elements_per_sample-1), elements_per_sample=i, mu_max_err=0.0, mu_max_err_global=mu_max_err_global, sigma_max_err=sigma_max_err, elements=None, use_percentages=False, cache_element_samples=True) for i in range(1, max_elements_per_sample)]

        X, y = reduce(lambda acc, val: (acc[0] + val[0], acc[1] + val[1]) , artificial_data, ([], []))

        return np.array(X), np.array(y)