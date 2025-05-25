import numpy as np
import random 

if __name__ == '__main__':
    from Elements import Elements
else:
    from .Elements import Elements
from functools import reduce
from tqdm import tqdm 
from scipy.stats import beta

import multiprocessing

class ArtifficialTrainDataGenerator2:
    CHANNELS_COUNT = 4096
    TARGET_VECTOR_LENGTH = len(Elements.LINES.keys())
    CACHED_ELEMENT_SAMPLES = {}
    SPECTRUM_PADDING = 196
    MIN_ENERGY = 0
    MAX_ENERGY = 20
    WOOD_SPECTRUM = np.load("data/objects/pigment_palette/1_calibrated_spectra.npz", allow_pickle=True)["wood"]

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
        return [el if el in Elements.LINES else Elements.NUM2SYMBOL[el] for el in selected_elements]

    @staticmethod
    def _add_peak(energy_range, element_line, element_sample, mu_err_global, mu_max_err, sigma, peak_type):
        """Add a specific peak (escape or copper) to the element sample."""
        if peak_type == "escape":
            mu = element_line["mu"] - Elements.ESCAPE_ENERGY_DIFF + np.random.uniform(-mu_max_err, mu_max_err) + mu_err_global
            intensity = element_line["intensity"] * Elements.ESCAPE_ENERGY_INTENSITY_RATIO
        # elif peak_type == "cu":
        #     mu = Elements.CU_THRESHOLD_ENERGY + np.random.uniform(-mu_max_err, mu_max_err) + mu_err_global
        #     intensity = element_line["intensity"] * Elements.ESCAPE_ENERGY_INTENSITY_RATIO
        else:
            raise ValueError("Invalid peak type")

        gaussian = ArtifficialTrainDataGenerator2.gaussian(energy_range, mu, sigma)
        gaussian /= np.max(gaussian) 
        gaussian *= intensity
        element_sample += gaussian
        return element_sample

    @staticmethod
    def _generate_element_sample(energy_range, element, mu_err_global, mu_max_err, sigma, cache_element_samples):
        """Generate a sample for a single element."""
        if element == "wood":
            return ArtifficialTrainDataGenerator2.WOOD_SPECTRUM.copy()


        if cache_element_samples and element in ArtifficialTrainDataGenerator2.CACHED_ELEMENT_SAMPLES:
            return ArtifficialTrainDataGenerator2.CACHED_ELEMENT_SAMPLES[element].copy()

        element_sample = np.zeros(ArtifficialTrainDataGenerator2.CHANNELS_COUNT)
        for element_line in Elements.get_parsed_element_lines(element):
            mu = element_line["mu"] + np.random.uniform(-mu_max_err, mu_max_err) + mu_err_global
            intensity = element_line["intensity"]
            gaussian = ArtifficialTrainDataGenerator2.gaussian(energy_range, mu, sigma)
            gaussian /= np.max(gaussian)
            gaussian *= intensity
            element_sample += gaussian

            # Add peaks if conditions met
            if mu > Elements.ESC_THRESHOLD_ENERGY:
                element_sample = ArtifficialTrainDataGenerator2._add_peak(energy_range      = energy_range
                                                              , element_line    = element_line
                                                              , element_sample  = element_sample
                                                              , mu_err_global   = mu_err_global
                                                              , mu_max_err      = mu_max_err
                                                              , sigma           = sigma
                                                              , peak_type       = "escape")
            
            # if mu > Elements.CU_THRESHOLD_ENERGY:
            #     element_sample = ArtifficialTrainDataGenerator2._add_peak(energy_range      = energy_range
            #                                                   , element_line    = element_line
            #                                                   , element_sample  = element_sample
            #                                                   , mu_err_global   = mu_err_global
            #                                                   , mu_max_err      = mu_max_err
            #                                                   , sigma_max_err   = sigma_max_err
            #                                                   , scale_sigma     = scale_sigma
            #                                                   , peak_type       = "cu")

        element_sample /= np.max(element_sample)

        if element not in ArtifficialTrainDataGenerator2.CACHED_ELEMENT_SAMPLES:
            ArtifficialTrainDataGenerator2.CACHED_ELEMENT_SAMPLES[element] = element_sample

        return element_sample

    @staticmethod
    def generate_sample(energy_range, selected_elements, element_percentages, mu_err_global, mu_max_err, sigma_range, set_percentages=False, cache_element_samples=False):
        """Generate a single sample with specified elements and parameters."""
        sample = np.zeros(ArtifficialTrainDataGenerator2.CHANNELS_COUNT)
        target = np.zeros(ArtifficialTrainDataGenerator2.TARGET_VECTOR_LENGTH + 1)

        for element_percentage, element in zip(element_percentages, selected_elements):
            target[Elements.SYMBOL2NUM[element]] = element_percentage if set_percentages else 1
            sigma = np.random.uniform(*sigma_range)
            element_sample = ArtifficialTrainDataGenerator2._generate_element_sample(energy_range    = energy_range
                                                                         , element                  = element
                                                                         , mu_err_global            = mu_err_global 
                                                                         , mu_max_err               = mu_max_err
                                                                         , sigma                    = sigma
                                                                         , cache_element_samples    = cache_element_samples)
            element_sample *= element_percentage
            element_sample[:ArtifficialTrainDataGenerator2.SPECTRUM_PADDING] = 0
            element_sample[-ArtifficialTrainDataGenerator2.SPECTRUM_PADDING:] = 0
            sample += element_sample
            target[Elements.SYMBOL2NUM[element]] = np.sum(element_sample)

        percent_wood_spectrum = random.random()
        wood_spectrum = percent_wood_spectrum * ArtifficialTrainDataGenerator2.WOOD_SPECTRUM 
        target *= (1 - percent_wood_spectrum)
        target[-1] = np.sum(wood_spectrum)
        sample = (1-percent_wood_spectrum) * sample + wood_spectrum 
        target /= np.max(sample)
        sample /= np.max(sample)

        return sample, target

    @staticmethod
    def generate_many_samples(samples=10, mu_max_err=0.0, mu_max_err_global=0.0, sigma_range=(0.2, 0.6), batch_size=5000, elements=None, set_percentages=False, cache_element_samples=False):
        """Generate multiple samples with specified parameters."""
        if samples <= batch_size:
            X = [np.zeros(ArtifficialTrainDataGenerator2.CHANNELS_COUNT) for _ in range(samples)]
            y = [np.zeros(ArtifficialTrainDataGenerator2.TARGET_VECTOR_LENGTH) for _ in range(samples)]
            energy_range = np.linspace(ArtifficialTrainDataGenerator2.MIN_ENERGY, ArtifficialTrainDataGenerator2.MAX_ENERGY, ArtifficialTrainDataGenerator2.CHANNELS_COUNT)

            
            spectra_counts = beta.rvs(a=4, b=20, size=samples)
            different_spectra_count = ArtifficialTrainDataGenerator2.TARGET_VECTOR_LENGTH
            spectra_counts = np.clip((spectra_counts * different_spectra_count).astype(int) + 1, 1, different_spectra_count).tolist()
            for i, spectra_count in enumerate(spectra_counts):
                selected_elements = ArtifficialTrainDataGenerator2._sample_elements(spectra_count, elements)
                element_percentages = ArtifficialTrainDataGenerator2._get_random_percentages(len(selected_elements))
                mu_err_global = np.random.uniform(-mu_max_err_global, mu_max_err_global)

                sample, target = ArtifficialTrainDataGenerator2.generate_sample(energy_range = energy_range
                                                                    , selected_elements     = selected_elements
                                                                    , element_percentages   = element_percentages
                                                                    , mu_err_global         = mu_err_global
                                                                    , mu_max_err            = mu_max_err
                                                                    , sigma_range           = sigma_range
                                                                    , set_percentages       = set_percentages
                                                                    , cache_element_samples = cache_element_samples)

                sample /= np.max(sample)
                X[i] = sample
                y[i] = target
            return X, y
        else:
            num_batches = samples // batch_size
            remainder = samples % batch_size

            batch_sizes = [batch_size] * num_batches + ([remainder] if remainder > 0 else [])

            with multiprocessing.Pool() as pool:
                results = list(tqdm(
                    pool.starmap(ArtifficialTrainDataGenerator2.generate_many_samples, [(s
                                                                             , mu_max_err
                                                                             , mu_max_err_global
                                                                             , sigma_range
                                                                             , batch_size
                                                                             , elements
                                                                             , set_percentages
                                                                             , cache_element_samples) for s in batch_sizes]),
                    total=len(batch_sizes)
                ))

            X, y = [], []
            for X_batch, y_batch in results:
                X.extend(X_batch)
                y.extend(y_batch)

            return X, y
    

    @staticmethod
    def generate_artificial_data(samples=10000, batch_size=5000, mu_max_err=0.05, mu_max_err_global=0.05, sigma_range=(0.2, 0.7), set_percentages=False, cache_element_samples=False):
        X, y= ArtifficialTrainDataGenerator2.generate_many_samples(samples = samples
                                                                        , mu_max_err            = mu_max_err
                                                                        , mu_max_err_global     = mu_max_err_global
                                                                        , sigma_range           = sigma_range 
                                                                        , batch_size            = batch_size
                                                                        , elements              = None
                                                                        , set_percentages       = set_percentages 
                                                                        , cache_element_samples = cache_element_samples) 

        return np.array(X), np.array(y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # x, y = ArtifficialTrainDataGenerator2.generate_sample(np.linspace(0, 20, 4096,), ["pb", "cu", "wood"], [1, 2, 0.5], 0, 0, (0.2, 0.7) )

    #    # plt.legend()
    # plt.savefig("temp.png")

    square_size = 20
    num_elements = len(Elements.LINES)
    energy_range = np.linspace(0, 20, 4096)

    result = np.zeros((num_elements * square_size, num_elements * square_size, 4096))

    for i, element_a in tqdm(enumerate(Elements.LINES.keys()), total=num_elements):
        for j, element_b in enumerate(Elements.LINES.keys()):
            for k in range(square_size):
                for m in range(square_size):
                    alpha = (square_size - 1 - k + m) / (2 * square_size)
                    alpha = np.clip(alpha, 0, 1)  # Ensure alpha stays in [0, 1]

                    a = ArtifficialTrainDataGenerator2._generate_element_sample(
                        energy_range, element_a,
                        mu_err_global=0.05, mu_max_err=0.05,
                        sigma=0.2, cache_element_samples=False
                    )
                    b = ArtifficialTrainDataGenerator2._generate_element_sample(
                        energy_range, element_b,
                        mu_err_global=0.05, mu_max_err=0.05,
                        sigma=0.2, cache_element_samples=False
                    )

                    result[square_size * i + k, square_size * j + m, :] = a * (1 - alpha) + b * alpha

    np.save("test_data.npy", result)




