import numpy as np
import random
from functools import reduce
from tqdm import tqdm
from scipy.stats import beta
import multiprocessing

try:
    from Elements import Elements
except Exception as e:
    from .Elements import Elements


class ArtificialTrainDataGenerator:
    CHANNELS = 4096
    TARGET_LEN = len(Elements.LINES)
    CACHE = {}
    PADDING = 196
    MIN_E = 0
    MAX_E = 20
    WOOD_SPECTRUM = np.load("data/objects/pigment_palette/pad_average_spectra_calibrated.npz", allow_pickle=True)["spectrum"].item()["wood"]

    @staticmethod
    def gaussian(x, mu, sigma):
        """Return Gaussian distribution for given mean and standard deviation."""
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    @staticmethod
    def random_percentages(n):
        """Return n random percentages summing to 1."""
        values = np.random.rand(n)
        return values / np.sum(values)

    @staticmethod
    def sample_elements(count, selected=None):
        """Randomly sample 'count' elements."""
        all_elements = list(Elements.LINES.keys())
        if selected:
            return [el if el in Elements.LINES else Elements.NUM2SYMBOL[el] for el in selected]
        return random.sample(all_elements, count)

    @staticmethod
    def add_peak(energy_range, line, spectrum, mu_global_err, mu_local_err, sigma, kind):
        """Add escape peak to the spectrum."""
        if kind != "escape":
            raise ValueError(f"Unknown peak type: {kind}")

        mu = line["mu"] - Elements.ESCAPE_ENERGY_DIFF + np.random.uniform(-mu_local_err, mu_local_err) + mu_global_err
        intensity = line["intensity"] * Elements.ESCAPE_ENERGY_INTENSITY_RATIO


        peak = ArtificialTrainDataGenerator.gaussian(energy_range, mu, sigma)
        peak = (peak / np.max(peak)) * intensity
        plt.plot(np.linspace(0, 20, 4096), peak, label=f"{kind}({line["name"]}) (μ={mu:.2f} keV, σ={sigma:.2f}keV)")
        return spectrum + peak

    @staticmethod
    def generate_element_sample(energy_range, element, mu_global_err, mu_local_err, sigma, use_cache):
        """Generate spectrum sample for a single element."""
        if element == "wood":
            return ArtificialTrainDataGenerator.WOOD_SPECTRUM.copy()

        if use_cache and element in ArtificialTrainDataGenerator.CACHE:
            return ArtificialTrainDataGenerator.CACHE[element].copy()

        spectrum = np.zeros(ArtificialTrainDataGenerator.CHANNELS)

        for line in Elements.get_parsed_element_lines(element):
            mu = line["mu"] + np.random.uniform(-mu_local_err, mu_local_err) + mu_global_err
            intensity = line["intensity"]

            peak = ArtificialTrainDataGenerator.gaussian(energy_range, mu, sigma)
            peak = (peak / np.max(peak)) * intensity
            plt.plot(np.linspace(0, 20, 4096), peak, label=f"{line["name"]} (μ={mu:.2f} keV, σ={sigma:.2f}keV)")
            spectrum += peak

            if mu > Elements.ESC_THRESHOLD_ENERGY:
                spectrum = ArtificialTrainDataGenerator.add_peak(
                    energy_range, line, spectrum, mu_global_err, mu_local_err, sigma, "escape"
                )

        spectrum /= np.max(spectrum)

        if use_cache:
            ArtificialTrainDataGenerator.CACHE[element] = spectrum.copy()

        return spectrum

    @staticmethod
    def generate_sample(energy_range, elements, percentages, mu_global_err, mu_local_err, sigma, use_cache=False):
        """Generate a single synthetic spectrum and its label."""
        sample = np.zeros(ArtificialTrainDataGenerator.CHANNELS)
        target = np.zeros(ArtificialTrainDataGenerator.TARGET_LEN + 1)

        for perc, element in zip(percentages, elements):
            spectrum = ArtificialTrainDataGenerator.generate_element_sample(
                energy_range, element, mu_global_err, mu_local_err, sigma, use_cache
            )

            spectrum *= perc
            spectrum[:ArtificialTrainDataGenerator.PADDING] = 0
            spectrum[-ArtificialTrainDataGenerator.PADDING:] = 0

            sample += spectrum
            target[Elements.SYMBOL2NUM[element]] = np.sum(spectrum)

        wood_ratio = np.random.beta(1.5, 5, size=1)[0]
        wood = wood_ratio * ArtificialTrainDataGenerator.WOOD_SPECTRUM
        sample = (1 - wood_ratio) * sample + wood
        target *= (1 - wood_ratio)
        target[-1] = np.sum(wood)

        sample /= np.max(sample)
        target /= np.max(sample)

        return sample, target

    @staticmethod
    def generate_many(samples, mu_local_err=0.0, mu_global_err=0.0, sigma_range=(0.2, 0.6),
                      batch_size=5000, elements=None, use_cache=False):
        """Generate many synthetic samples."""
        if samples <= batch_size:
            X, y = [], []
            energy_range = np.linspace(ArtificialTrainDataGenerator.MIN_E, ArtificialTrainDataGenerator.MAX_E, ArtificialTrainDataGenerator.CHANNELS)
            counts = beta.rvs(a=4, b=20, size=samples)
            counts = np.clip((counts * ArtificialTrainDataGenerator.TARGET_LEN).astype(int) + 1, 1, ArtificialTrainDataGenerator.TARGET_LEN)

            for count in tqdm(counts, desc="Generating Samples"):
                selected = ArtificialTrainDataGenerator.sample_elements(count, elements)
                percentages = ArtificialTrainDataGenerator.random_percentages(len(selected))
                mu_err = np.random.uniform(-mu_global_err, mu_global_err)
                sigma = np.random.uniform(*sigma_range)

                sample, label = ArtificialTrainDataGenerator.generate_sample(
                    energy_range, selected, percentages, mu_err, mu_local_err, sigma, use_cache
                )

                X.append(sample)
                y.append(label)

            return X, y

        # Use multiprocessing for large batches
        batch_sizes = [batch_size] * (samples // batch_size)
        if samples % batch_size:
            batch_sizes.append(samples % batch_size)

        args = [(size, mu_local_err, mu_global_err, sigma_range, batch_size, elements, use_cache) for size in batch_sizes]

        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.starmap(ArtificialTrainDataGenerator.generate_many, args), total=len(args)))

        X, y = [], []
        for x_batch, y_batch in results:
            X.extend(x_batch)
            y.extend(y_batch)

        return X, y

    @staticmethod
    def generate_dataset(samples=10000, batch_size=5000, mu_local_err=0.05, mu_global_err=0.05,
                         sigma_range=(0.2, 0.7), use_cache=False):
        """Convenience method for generating a dataset."""
        X, y = ArtificialTrainDataGenerator.generate_many(
            samples=samples,
            mu_local_err=mu_local_err,
            mu_global_err=mu_global_err,
            sigma_range=sigma_range,
            batch_size=batch_size,
            elements=None,
            use_cache=use_cache
        )
        return np.array(X), np.array(y)


# Visualization or debugging
if __name__ == "__main__":
    import matplotlib.pyplot as plt


    spectrum = ArtificialTrainDataGenerator.generate_element_sample(
        energy_range=np.linspace(0, 20, ArtificialTrainDataGenerator.CHANNELS),
        element="au",
        mu_global_err=0.05,
        mu_local_err=0.05,
        sigma=0.6,
        use_cache=False)
    plt.plot(np.linspace(0, 20, 4096), spectrum, label=r"Mixture of Au Lines", color="black")
    
    plt.title("Sample Spectrum of Au")
    plt.ylim(0, 1.5)
    plt.xlabel("Energy (keV)")
    plt.ylabel("Intensity (a. u.)")
    plt.legend()
    plt.savefig("data/plots/au.svg")


    plt.figure()
    x, y = ArtificialTrainDataGenerator.generate_sample(
        energy_range=np.linspace(0, 20, ArtificialTrainDataGenerator.CHANNELS),
        elements=["au"],
        percentages=[1.0],
        mu_local_err=0.05,
        mu_global_err=0.1,
        sigma=0.6,
        use_cache=False)
    plt.figure()
    plt.plot(np.linspace(0, 20, 4096), spectrum, label=r"Mixture of Au Lines", color="black")
    plt.plot(np.linspace(0, 20, 4096), x, label="Au Spectrum with Background Noise", color="blue")

    plt.title("Spectrum of Au with Background Noise")
    plt.ylim(0, 1.5)
    plt.xlabel("Energy (keV)")
    plt.ylabel("Intensity (a. u.)")
    plt.legend()
    plt.savefig("data/plots/au_noise.svg")

    from FeatureEnhancer import FeatureEnhancer 
    plt.figure()
    plt.plot(np.linspace(0, 20, 4096), FeatureEnhancer._metropolis_hasting_mcmc(x, steps=20000), label="Au Spectrum Augmented with Metropolis-Hastings", color="red")
    plt.plot(np.linspace(0, 20, 4096), x, label=r"Au Spectrum with Background Noise", color="black")
    plt.title("Au Spectrum Augmented with Metropolis-Hastings")
    plt.ylim(0, 1.5)
    plt.xlabel("Energy (keV)")
    plt.ylabel("Intensity (a. u.)")
    plt.legend()
    plt.savefig("data/plots/au_mcmc.svg")


    # grid_size = 20
    # num_elements = len(Elements.LINES)
    # energy_range = np.linspace(0, 20, 4096)
    # result = np.zeros((num_elements * grid_size, num_elements * grid_size, 4096))

    # for i, a in tqdm(enumerate(Elements.LINES), total=num_elements, desc="Left Element"):
    #     for j, b in enumerate(Elements.LINES):
    #         for k in range(grid_size):
    #             for m in range(grid_size):
    #                 alpha = (grid_size - 1 - k + m) / (2 * grid_size)
    #                 alpha = np.clip(alpha, 0, 1)

    #                 spec_a = ArtificialTrainDataGenerator.generate_element_sample(
    #                     energy_range, a, mu_global_err=0.05, mu_local_err=0.05, sigma=0.6, use_cache=False
    #                 )
    #                 spec_b = ArtificialTrainDataGenerator.generate_element_sample(
    #                     energy_range, b, mu_global_err=0.05, mu_local_err=0.05, sigma=0.6, use_cache=False
    #                 )

    #                 result[i * grid_size + k, j * grid_size + m] = spec_a * (1 - alpha) + spec_b * alpha
    #                 result[i * grid_size + k, j * grid_size + m] /= np.max(result[i * grid_size + k, j * grid_size + m])

    # np.save("data/objects/test_data_default.npy", result)