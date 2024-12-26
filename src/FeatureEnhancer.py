from concurrent.futures import ProcessPoolExecutor, as_completed
from random import randint, gauss, uniform
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d, uniform_filter1d
from tqdm import tqdm
import numpy as np

class FeatureEnhancer():
    @staticmethod
    def _reduced_dynamic_range(sample, kernel_size):
        previous_shape = sample.shape
        sample = sample.squeeze()
        max_sample = maximum_filter1d(sample, kernel_size, mode="nearest")
        min_sample = minimum_filter1d(sample, kernel_size, mode="nearest")
        max_sample = gaussian_filter1d(max_sample, kernel_size, mode="nearest")
        min_sample = gaussian_filter1d(min_sample, kernel_size, mode="nearest")

        result = (sample - min_sample) / (max_sample - min_sample + 10e-10)

        result[result < 0] = 0
        result[:32] = result[32]
        result[-32:] = result[-32]
        result = (result - np.min(result)) / (np.max(result) - np.min(result) + 10e-10)
        return result.reshape(previous_shape)

    @staticmethod
    def _metropolis_hasting_mcmc(spectrum, steps=5000, sigma=256):
        channel_count = spectrum.shape[0]
        result = np.zeros_like(spectrum)

        x_prev = randint(0, channel_count - 1)
        probability_prev = spectrum[x_prev]

        for i in range(steps):
            x_proposed = int(gauss(x_prev, sigma)) % channel_count
            probability_proposed = spectrum[x_proposed]

            accept_criterion = probability_proposed / probability_prev
            accept_threshold = uniform(0, 1)
            if accept_criterion > accept_threshold:
                x_prev = x_proposed
                probability_prev = probability_proposed

            result[x_prev] += 1

        return result / np.max(result)

    @staticmethod
    def process_spectrum(spectrum, kernel_sizes, use_mcmc=False):
        num_kernels = len(kernel_sizes)

        if use_mcmc:
            spectrum = FeatureEnhancer._metropolis_hasting_mcmc(spectrum, steps=randint(5000, 30000))

        smoothened_spectra = np.empty((num_kernels, spectrum.shape[0]))
        reduced_dynamic_range_spectra = np.empty((num_kernels, spectrum.shape[0]))

        for j, kernel_size in enumerate(kernel_sizes):
            smoothened = gaussian_filter1d(spectrum, kernel_size)
            reduced_dynamic_range = FeatureEnhancer._reduced_dynamic_range(smoothened, kernel_size)

            smoothened_spectra[j] = smoothened / (np.max(smoothened) + 10e-10)
            reduced_dynamic_range_spectra[j] = reduced_dynamic_range / (np.max(reduced_dynamic_range) + 10e-10)

        return np.stack([smoothened_spectra, reduced_dynamic_range_spectra], axis=0)

    @staticmethod
    def enhanced_features(X, use_mcmc=False):
        kernel_sizes = [i for i in range(128)]
        num_channels = 2

        result = np.zeros((X.shape[0], num_channels, len(kernel_sizes), X.shape[1]))

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(FeatureEnhancer.process_spectrum, spectrum, kernel_sizes, use_mcmc): i for i, spectrum in enumerate(X)}

            for future in tqdm(as_completed(futures), desc="enhancing data features", total=len(X)):
                i = futures[future]
                result[i] = future.result()

        return result