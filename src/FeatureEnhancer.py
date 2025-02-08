from random import randint, gauss, uniform
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
from tqdm import tqdm
import numpy as np
import multiprocessing

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

    import numpy as np

    @staticmethod
    def process_spectrum(spectrum):
        steps = randint(500, 50000)

        spectrum = FeatureEnhancer._metropolis_hasting_mcmc(spectrum, steps=steps)

        return spectrum

    @staticmethod
    def _process_spectra(spectra: np.array, order: int):
        for i in range(len(spectra)):
            spectra[i] = FeatureEnhancer.process_spectrum(spectra[i])
        return spectra, order

    @staticmethod
    def process_spectra(spectra: np.array, batch_size=5000):
        batches = np.array_split(spectra, spectra.shape[0] // batch_size + 1)

        with multiprocessing.Pool() as pool:
            results = list(tqdm(
                pool.starmap(FeatureEnhancer._process_spectra, [(batch, i) for i, batch in enumerate(batches)]),
                total=len(batches)))

        X = []
        for X_batch, i in results:
            X.extend(X_batch)

        return np.array(X) 

if __name__ == "__main__":
    FeatureEnhancer.process_spectra(np.random.poiss (1000, 4096))