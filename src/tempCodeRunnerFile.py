    au= ArtificialTrainDataGenerator.generate_element_sample(
        energy_range=np.linspace(0, 20, ArtificialTrainDataGenerator.CHANNELS),
        element="au",
        mu_global_err=0.05,
        mu_local_err=0.05,
        sigma=0.6,
        use_cache=False)