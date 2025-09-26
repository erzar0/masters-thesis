import numpy as np
import random
from functools import reduce
from tqdm import tqdm
from scipy.stats import beta
import multiprocessing

try:
    from Elements import Elements
    from FeatureEnhancer import FeatureEnhancer
except Exception as e:
    from .Elements import Elements
    from .FeatureEnhancer import FeatureEnhancer


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
    def cumsum_window(
        spectrum: np.ndarray,
        left_max: float = 1.0,
        right_min: float = 0.0
    ) -> np.ndarray:
        cumsum = np.cumsum(spectrum[::-1])[::-1]
        cumsum = cumsum / np.max(cumsum)

        return right_min + cumsum * (left_max - right_min)

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
        name = line["name"].split("_")
        name[0] = name[0].capitalize()
        name[1] = name[1].capitalize()
        name[2] = name[2].replace("alpha", "α").replace("beta", "β")
        name = name[0] + " " + name[1] + name[2]
        plt.plot(np.linspace(0, 20, 4096), peak, label=f"Escape {name} (μ={mu:.2f} keV, σ={sigma:.2f}keV)")
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
            name = line["name"].split("_")
            name[0] = name[0].capitalize()
            name[1] = name[1].capitalize()
            name[2] = name[2].replace("alpha", "α").replace("beta", "β")
            name = name[0] + " " + name[1] + name[2]
            plt.plot(np.linspace(0, 20, 4096), peak, label=f"{name} (μ={mu:.2f} keV, σ={sigma:.2f}keV)")
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
        

        wood_spectrum = ArtificialTrainDataGenerator.WOOD_SPECTRUM.copy()

        left_max = np.random.uniform(0.0, 1)
        right_min = np.random.uniform(0.0, left_max)
        cumsum_window = ArtificialTrainDataGenerator.cumsum_window(wood_spectrum, left_max=left_max, right_min=right_min)

        background_spectrum = wood_spectrum * cumsum_window

        background_ratio = np.random.uniform(0, 1)
        while True:
            attenuated_background_spectrum = background_spectrum * background_ratio 
            attenuated_sample = (1 - background_ratio) * sample
            attenuated_target = (1 - background_ratio) * target
            attenuated_target[-1] = np.sum(attenuated_background_spectrum)
            final_sample = attenuated_sample + attenuated_background_spectrum

            division_coeff = np.max(final_sample)
            normalized_attenuated_background_spectrum = attenuated_background_spectrum / division_coeff
            if np.any(normalized_attenuated_background_spectrum > wood_spectrum): 
                background_ratio /= 2
                continue
            
            final_target = attenuated_target / np.max(final_sample)
            final_sample = final_sample / np.max(final_sample)
            break


        return final_sample, final_target 

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

    # x, y= ArtificialTrainDataGenerator.generate_element_sample(
    #     energy_range=np.linspace(0, 20, ArtificialTrainDataGenerator.CHANNELS),
    #     element="au",
    #     percentages=[0.6, 0.4],
    #     mu_global_err=0.05,
    #     mu_local_err=0.05,
    #     sigma=0.6,
    #     use_cache=False
    # )

    # x = FeatureEnhancer.add_poisson_noise(x, scale=1)
    # plt.plot(x)
    # print(y)
    # plt.savefig("sample_spectrum.svg", bbox_inches='tight', pad_inches=0.0)


    au= ArtificialTrainDataGenerator.generate_element_sample(
        energy_range=np.linspace(0, 20, ArtificialTrainDataGenerator.CHANNELS),
        element="au",
        mu_global_err=0.05,
        mu_local_err=0.05,
        sigma=0.6,
        use_cache=False)
    
    plt.plot(np.linspace(0, 20, 4096), au, label=r"Mixture of Au Spectral Lines", color="black", alpha=1)
    plt.ylim(0, 1.5)
    plt.legend()
    plt.ylabel("Intensity (a. u.)")
    plt.xlabel("Energy (keV)")
    plt.title("Sample Spectrum of Gold (Au)")
    plt.savefig("data/plots/au.svg", bbox_inches='tight', pad_inches=0.1)

    # fe = ArtificialTrainDataGenerator.generate_element_sample(
    #     energy_range=np.linspace(0, 20, ArtificialTrainDataGenerator.CHANNELS),
    #     element="fe",
    #     mu_global_err=0.05,
    #     mu_local_err=0.05,
    #     sigma=0.6,
    #     use_cache=False)
    
    # plt.figure()
    # au = au * 0.6
    # fe = fe * 0.4
    # res= au + fe
    # # res /= np.max(res)
    # plt.plot(np.linspace(0, 20, 4096), au, label=r"Au Spectrum", color="green", alpha=1)
    # plt.plot(np.linspace(0, 20, 4096), fe, label=r"Fe Spectrum", color="red", alpha=1)
    # plt.plot(np.linspace(0, 20, 4096), res, label=r"Fe-Au Mixture", color="black", alpha=1, linestyle='dotted')
    # plt.title("Mixture of Au and Fe Spectra")
    # plt.legend()
    # plt.xlabel("Energy (keV)")
    # plt.ylabel("Intensity (a. u.)")
    # plt.savefig("data/plots/au_fe_mixture.svg", bbox_inches='tight', pad_inches=0.0)



    # spectrum = ArtificialTrainDataGenerator.generate_element_sample(
    #     energy_range=np.linspace(0, 20, ArtificialTrainDataGenerator.CHANNELS),
    #     element="au",
    #     mu_global_err=0.05,
    #     mu_local_err=0.05,
    #     sigma=0.6,
    #     use_cache=False)
    # plt.plot(np.linspace(0, 20, 4096), spectrum, label=r"Mixture of Au Lines", color="black")
    
    # plt.title("Artificial Spectrum of Au")
    # plt.ylim(0, 1.5)
    # plt.xlabel("Energy (keV)")
    # plt.ylabel("Intensity (a. u.)")
    # plt.legend()
    # plt.savefig("data/plots/au.svg")


    # plt.figure()
    # x = ArtificialTrainDataGenerator.generate_element_sample(

    #     energy_range=np.linspace(0, 20, ArtificialTrainDataGenerator.CHANNELS),
    #     element="au",
    #     mu_local_err=0.05,
    #     mu_global_err=0.1,
    #     sigma=0.6,
    #     use_cache=False)
    # plt.figure()
    # window = ArtificialTrainDataGenerator.cumsum_window(x, left_max=0.8, right_min=0.2)
    # wood_spectrum = ArtificialTrainDataGenerator.WOOD_SPECTRUM * window

    # fig, axs = plt.subplots(1, 2, figsize=(12, 4))


    # # --- First Subplot ---
    # coeff = np.max(x * 0.6 + wood_spectrum * 0.4)
    # axs[0].plot(np.linspace(0, 20, 4096), x * 0.6 / coeff, label=r"Au Spectrum", color="blue")
    # axs[0].plot(np.linspace(0, 20, 4096), wood_spectrum * 0.4 / coeff, label="Attenuated Background Spectrum", color="red")
    # res = x * 0.6 + wood_spectrum * 0.4
    # axs[0].plot(np.linspace(0, 20, 4096), res / coeff, label="Au Spectrum Augmented with Background", color="black", linestyle="dotted")
    # axs[0].set_title("Spectrum of Au Augmented with Background")
    # axs[0].set_ylim(0, 1.3)
    # axs[0].set_xlabel("Energy (keV)")
    # axs[0].set_ylabel("Intensity (a. u.)")
    # axs[0].legend()

    # # --- Second Subplot ---
    # axs[1].plot(np.linspace(0, 20, 4096), FeatureEnhancer.add_poisson_noise(res, scale=5), 
    #             label="Au Spectrum Augmented with Poisson Noise", color="red")
    # axs[1].plot(np.linspace(0, 20, 4096), res / np.max(res), 
    #             label=r"Au Spectrum Augmented with Background", color="black")
    # axs[1].set_title("Spectrum of Au Augmented with Poisson Noise")
    # axs[1].set_ylim(0, 1.3)
    # axs[1].set_xlabel("Energy (keV)")
    # axs[1].set_ylabel("Intensity (a. u.)")
    # axs[1].legend()


    # plt.tight_layout()

    # plt.savefig("data/plots/au_combined.svg")

    # plt.figure(figsize=(10, 5))
    # # e = np.linspace(0, 20, 4096)
    # e = np.arange(4096)
    # # plt.plot(e, wood_spectrum * 0.4 / coeff, label="Attenuated Background Spectrum", color="red", alpha=0.5)
    # plt.fill_between(e, 0, wood_spectrum * 0.4 / coeff, color="red", alpha=0.3, label=f"Attenuated Background Spectrum Area = {np.sum(wood_spectrum * 0.4 / coeff):.2f}")
    # # plt.plot(e, res / coeff, label=r"Au Spectrum", color="blue", alpha=0.5)
    # plt.fill_between(e, wood_spectrum * 0.4 / coeff, res / coeff, color="blue", alpha=0.3, label=f"Au Spectrum Area = {np.sum(x * 0.6 / coeff):.2f}")
    # plt.plot(e, res / coeff, label="Au Spectrum Augmented with Background", color="black")
    # plt.ylabel("Intensity (a. u.)")
    # plt.xlabel("Energy (a. u.)")
    # plt.legend(loc="upper left")
    # plt.savefig("data/plots/au_stacked.svg", bbox_inches='tight', pad_inches=0.1)

    # from matplotlib.animation import FuncAnimation
    # import numpy as np

    # coeff = 1.0

    # # Precompute spectra
    # bg = wood_spectrum * 0.4 / coeff
    # au_final = res / coeff
    # au_only = au_final - bg  # Au contribution above background

    # # Areas (consistent with static plot)
    # bg_area = np.sum(bg)
    # au_area = np.sum(au_final - bg)

    # # Create figure
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.set_ylabel("Intensity (a. u.)")
    # ax.set_xlabel("Energy (a.u)")

    # bg_label = f"Attenuated Background Spectrum Area"
    # au_label = f"Au Spectrum Area"

    # (line,) = ax.plot([], [], color="black",
    #                 label="Au Spectrum Augmented with Background")
    # bg_patch = ax.fill_between([], [], [], color="red", alpha=0.3, label=bg_label)
    # au_patch = ax.fill_between([], [], [], color="blue", alpha=0.3, label=au_label)
    # ax.legend(loc="upper left")

    # # Remove placeholders
    # bg_patch.remove()
    # au_patch.remove()
    # bg_patch = None
    # au_patch = None

    # def update(frame):
    #     global bg_patch, au_patch
    #     if bg_patch: bg_patch.remove()
    #     if au_patch: au_patch.remove()

    #     if frame < 50:  # Stage 1: pure Au only
    #         bg_patch = ax.fill_between(e, 0, 0, color="red", alpha=0.3)
    #         au_patch = ax.fill_between(e, 0, au_only, color="blue", alpha=0.3)
    #         line.set_data([], [])
    #     elif frame < 100:  # Stage 2: red grows in, blue morphs
    #         frac = (frame - 50) / 50
    #         bg_patch = ax.fill_between(e, 0, bg * frac, color="red", alpha=0.3)
    #         blue_bottom = frac * bg
    #         blue_top = (1 - frac) * au_only + frac * au_final
    #         au_patch = ax.fill_between(e, blue_bottom, blue_top, color="blue", alpha=0.3)
    #         line.set_data([], [])
    #     else:  # Stage 3: final state
    #         bg_patch = ax.fill_between(e, 0, bg, color="red", alpha=0.3)
    #         au_patch = ax.fill_between(e, bg, au_final, color="blue", alpha=0.3)
    #         line.set_data(e, au_final)

    #     return [bg_patch, au_patch, line]

    # ani = FuncAnimation(fig, update, frames=150, interval=50, blit=False)

    # # Save as MP4 (requires ffmpeg)
    # ani.save("data/plots/au_stacked_animation.mp4", writer="ffmpeg", dpi=300)






    # ba = ArtificialTrainDataGenerator.generate_element_sample(
    #     energy_range=np.linspace(0, 20, ArtificialTrainDataGenerator.CHANNELS),
    #     element="ba",
    #     mu_global_err=0.05,
    #     mu_local_err=0.05,
    #     sigma=0.6,
    #     use_cache=False
    # )
    # ti = ArtificialTrainDataGenerator.generate_element_sample(
    #     energy_range=np.linspace(0, 20, ArtificialTrainDataGenerator.CHANNELS),
    #     element="ti",
    #     mu_global_err=0.05,
    #     mu_local_err=0.05,
    #     sigma=0.6,
    #     use_cache=False)
    
    # plt.figure(figsize=(12, 2))
    # plt.plot(np.linspace(0, 20, 4096), ba, label=r"Ba Spectrum", color="blue")
    # plt.plot(np.linspace(0, 20, 4096),ti, label=r"Ti Spectrum", color="orange")
    # plt.title("Spectra of Ba and Ti")
    # plt.ylim(0, 1.0)
    # plt.xlabel("Energy (keV)")
    # plt.ylabel("Intensity (a. u.)")
    # plt.legend()
    # plt.savefig("data/plots/ba_ti.svg", bbox_inches='tight', pad_inches=0.1)

    # plt.figure(figsize=(12, 2))

    # ba = ArtificialTrainDataGenerator.generate_element_sample(
    #     energy_range=np.linspace(0, 20, ArtificialTrainDataGenerator.CHANNELS),
    #     element="k",
    #     mu_global_err=0.05,
    #     mu_local_err=0.05,
    #     sigma=0.6,
    #     use_cache=False
    # )
    # ti = ArtificialTrainDataGenerator.generate_element_sample(
    #     energy_range=np.linspace(0, 20, ArtificialTrainDataGenerator.CHANNELS),
    #     element="cd",
    #     mu_global_err=0.05,
    #     mu_local_err=0.05,
    #     sigma=0.6,
    #     use_cache=False)
    
    # plt.plot(np.linspace(0, 20, 4096),ba, label=r"K Spectrum", color="blue")
    # plt.plot(np.linspace(0, 20, 4096),ti, label=r"Cd Spectrum", color="orange")
    # plt.title("Spectra of K and Cd")
    # plt.ylim(0, 1.0)
    # plt.xlabel("Energy (keV)")
    # plt.ylabel("Intensity (a. u.)")
    # plt.legend()
    # plt.savefig("data/plots/k_cd.svg", bbox_inches='tight', pad_inches=0.1)






    # grid_size = 20
    # num_elements = len(Elements.LINES)
    # energy_range = np.linspace(0, 20, 4096)
    # result = np.zeros((num_elements * grid_size, num_elements * grid_size, 4096))

    # for i, a in enumerate(Elements.LINES):
    #     for j, b in tqdm(enumerate(Elements.LINES)):
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

    #                 sample = spec_a * (1 - alpha) + spec_b * alpha
    #                 sample /= np.max(sample)
    #                 # result[i * grid_size + k, j * grid_size + m] = 
    #                 # result[i * grid_size + k, j * grid_size + m] /= np.max(result[i * grid_size + k, j * grid_size + m])

    #                 wood_spectrum = ArtificialTrainDataGenerator.WOOD_SPECTRUM.copy()

    #                 left_max = np.random.uniform(0.0, 1)
    #                 right_min = np.random.uniform(0.0, left_max)
    #                 cumsum_window = ArtificialTrainDataGenerator.cumsum_window(wood_spectrum, left_max=left_max, right_min=right_min)

    #                 background_spectrum = wood_spectrum * cumsum_window

                    
    #                 background_ratio = np.random.uniform(0, 1)
    #                 while True:
    #                     attenuated_background_spectrum = background_spectrum * background_ratio 
    #                     attenuated_sample = (1 - background_ratio) * sample
    #                     final_sample = attenuated_sample + attenuated_background_spectrum

    #                     division_coeff = np.max(final_sample)
    #                     normalized_attenuated_background_spectrum = attenuated_background_spectrum / division_coeff
    #                     if np.any(normalized_attenuated_background_spectrum > wood_spectrum): 
    #                         background_ratio /= 2
    #                         continue
                        
    #                     final_sample = final_sample / np.max(final_sample)
    #                     break

    #                 result[i * grid_size + k, j * grid_size + m] = sample
    
    # h, w, _ = result.shape
    # result = FeatureEnhancer.process_spectra(result.reshape((h * w, -1))).reshape(h, w, -1)
    
    # np.save("data/objects/pigment_vs_pigment_noise.npy", result)