from random import randint, gauss, uniform
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
from tqdm import tqdm
import numpy as np
import multiprocessing

class FeatureEnhancer:
    @staticmethod
    def process_spectra(spectra: np.array, batch_size=5000):
        """
        Processes a batch of spectra by applying a series of enhancement steps
        in parallel using multiprocessing.

        Parameters:
            spectra (np.array): A 2D NumPy array where each row represents a spectrum
                                to be processed. Expected shape: (num_spectra, spectrum_length).
            batch_size (int, optional): The number of spectra to process in each batch.
                                        Larger batch sizes can reduce overhead but increase
                                        memory usage per process. Defaults to 5000.

        Returns:
            np.array: A 2D NumPy array containing the processed spectra,
                      with the same shape as the input `spectra`.
        """
        batches = np.array_split(spectra, spectra.shape[0] // batch_size + 1)

        with multiprocessing.Pool() as pool:
            results = list(tqdm(
                pool.starmap(FeatureEnhancer._process_spectra_batch, [(batch, i) for i, batch in enumerate(batches)]),
                total=len(batches)))

        # Reconstruct the processed spectra in their original order
        X = [None] * len(batches)
        for X_batch, i in results:
            X[i] = X_batch

        return np.vstack(X)

    @staticmethod
    def process_spectrum(spectrum: np.array):
        """
        Applies a series of enhancement steps to a single spectrum.

        Parameters:
            spectrum (np.array): A 1D NumPy array representing a single spectrum
                                 to be processed. Expected shape: (spectrum_length,).

        Returns:
            np.array: A 1D NumPy array representing the enhanced spectrum,
                      with the same shape as the input `spectrum`.
        """
        steps = randint(500, 100000)
        spectrum = FeatureEnhancer._metropolis_hasting_mcmc(spectrum, steps=steps)
        return spectrum

    @staticmethod
    def _reduced_dynamic_range(sample: np.array, kernel_size: int):
        """
        Reduces the dynamic range of a sample (e.g., a spectrum or a segment)
        by normalizing values based on local minimum and maximum, then smoothing.

        Parameters:
            sample (np.array): The input array (typically a 1D spectrum or a segment of it)
                               to which dynamic range reduction will be applied.
                               Expected shape: (length,).
            kernel_size (int): The size of the kernel used for the `maximum_filter1d`,
                               `minimum_filter1d`, and `gaussian_filter1d` operations.
                               Determines the local window for min/max calculation and
                               smoothing. Should be a positive integer.

        Returns:
            np.array: A NumPy array with reduced dynamic range, normalized between 0 and 1,
                      and reshaped to the original input `sample`'s shape.
        """
        previous_shape = sample.shape
        sample = sample.squeeze()  # Ensure 1D for processing

        max_sample = maximum_filter1d(sample, kernel_size, mode="nearest")
        min_sample = minimum_filter1d(sample, kernel_size, mode="nearest")

        max_sample = gaussian_filter1d(max_sample, kernel_size, mode="nearest")
        min_sample = gaussian_filter1d(min_sample, kernel_size, mode="nearest")

        # Normalize the sample to the [0, 1] range based on local min/max.
        # Add a small epsilon to the denominator to prevent division by zero.
        result = (sample - min_sample) / (max_sample - min_sample + 1e-10)

        # Clip values below 0 to 0.
        result[result < 0] = 0

        # Extend boundary values to avoid edge effects or artifacts at the ends of the spectrum.
        result[:32] = result[32]
        result[-32:] = result[-32]

        # Re-normalize the entire result to a [0, 1] range.
        # Add a small epsilon to the denominator to prevent division by zero.
        result = (result - np.min(result)) / (np.max(result) - np.min(result) + 1e-10)
        return result.reshape(previous_shape)

    @staticmethod
    def _metropolis_hasting_mcmc(spectrum: np.array, steps: int = 5000, sigma: int = 256):
        """
        Applies the Metropolis-Hastings Markov Chain Monte Carlo (MCMC) algorithm
        to a spectrum, treating its values as probabilities. This effectively
        redistributes the "intensity" of the spectrum based on a probabilistic walk.

        Parameters:
            spectrum (np.array): The input spectrum (1D NumPy array) to apply
                                 Metropolis-Hastings MCMC sampling to. This array
                                 is treated as a probability distribution where
                                 higher values indicate higher likelihoods of being sampled.
            steps (int, optional): The number of MCMC steps (iterations) to perform.
                                   More steps lead to a more refined sampling and a
                                   smoother output but increase computation time. Defaults to 5000.
            sigma (int, optional): The standard deviation for the Gaussian proposal
                                   distribution used to generate new candidate positions.
                                   A larger `sigma` allows for larger jumps in the sample space,
                                   facilitating broader exploration, while a smaller `sigma`
                                   promotes local exploration. Defaults to 256.

        Returns:
            np.array: A 1D NumPy array representing the sampled spectrum,
                      normalized to a [0, 1] range.
        """
        channel_count = spectrum.shape[0]
        result = np.zeros_like(spectrum, dtype=float)  # Use float for accumulation

        # Initialize the current position and its probability.
        x_prev = randint(0, channel_count - 1)
        probability_prev = spectrum[x_prev]

        for _ in range(steps):
            # Propose a new position using a Gaussian distribution centered at x_prev.
            x_proposed = int(gauss(x_prev, sigma)) % channel_count
            
            # Ensure x_proposed is within valid array bounds.
            if not (0 <= x_proposed < channel_count):
                continue

            probability_proposed = spectrum[x_proposed]

            # Skip if proposed probability is non-positive to avoid division by zero or
            # issues with acceptance ratio calculation (e.g., log of non-positive).
            if probability_proposed <= 0:
                continue

            # Metropolis-Hastings acceptance criterion:
            # accept_criterion = min(1, P(x_proposed) / P(x_prev))
            # Add a small epsilon to the denominator to prevent division by zero.
            accept_criterion = probability_proposed / (probability_prev + 1e-5)
            accept_threshold = uniform(0, 1)

            # Accept or reject the proposed step. If accepted, update the current position.
            if accept_criterion > accept_threshold:
                x_prev = x_proposed
                probability_prev = probability_proposed

            # Increment the count for the current accepted position. This effectively builds a
            # histogram of visited states, representing the sampled distribution.
            result[x_prev] += 1

        # Normalize the result to a [0, 1] range based on the maximum count.
        max_result = np.max(result)
        if max_result == 0:  # Handle case where no points were accepted (e.g., all probabilities were zero)
            return result
        return result / max_result

    @staticmethod
    def _process_spectra_batch(spectra_batch: np.array, order: int):
        """
        Internal helper method to process a single batch of spectra.
        Designed to be used with multiprocessing.

        Parameters:
            spectra_batch (np.array): A 2D NumPy array representing a batch of spectra
                                      to be processed. Expected shape: (batch_size, spectrum_length).
            order (int): An integer representing the original order or index of this batch
                         within the larger set of spectra. Used to reassemble results
                         correctly after parallel processing.

        Returns:
            tuple: A tuple containing:
                   - np.array: The processed batch of spectra, with the same shape as `spectra_batch`.
                   - int: The `order` of the batch, returned for reassembly.
        """
        processed_spectra = np.zeros_like(spectra_batch, dtype=float)
        for i in range(len(spectra_batch)):
            processed_spectra[i] = FeatureEnhancer.process_spectrum(spectra_batch[i])
        return processed_spectra, order


if __name__ == "__main__":
    FeatureEnhancer.process_spectra(np.random.poiss (1000, 4096))