import numpy as np
from collections import OrderedDict


class Elements():
    LINES = OrderedDict()
    LINES["as"] = (("k_alpha", 10.523, (1**2 + 0.51**2)**0.5), ("k_beta31", 11.723, (0.06**2 + 0.13**2)**0.5))
    LINES["au"] = (("l_alpha", 9.700, (1**2 + 0.11**2)**0.5), ("l_beta", 11.442, 0.55))
    LINES["ba"] = (("l_alpha", 4.460, (1**2 + 0.11**2)**0.5), ("l_beta", 4.82, 0.6))
    LINES["ca"] = (("k_alpha", 3.690, (1**2 + 0.5**2)**0.5), ("k_beta", 4.012, 0.13))
    LINES["cd"] = (("l_alpha", 3.130, (1**2 + 0.11**2)**0.5), ("l_beta1", 3.316, 0.58), ("l_beta", 3.528, 0.15), ("l_gamma", 3.716, 0.06))
    LINES["co"] = (("k_alpha", 6.927, (1**2 + 0.51**2)**0.5), ("k_beta", 7.649, 0.17))
    LINES["cr"] = (("k_alpha", 5.410, (1**2 + 0.5**2)**0.5), ("k_beta", 5.946, 0.15))
    LINES["cu"] = (("k_alpha", 8.037, (1**2 + 0.51**2)**0.5), ("k_beta", 8.905, 0.17))
    LINES["fe"] = (("k_alpha", 6.395, (1**2 + 0.5**2)**0.5), ("k_beta", 7.058, 0.17))
    LINES["hg"] = (("l_alpha2", 9.897, 0.11), ("l_alpha1", 9.988, 1), ("l_beta1", 11.824, 0.65))
    LINES["k" ] = (("k_alpha", 3.314, (1**2 + 0.5**2)**0.5), ("k_beta", 3.589, 0.11))
    LINES["mn"] = (("k_alpha", 5.893, (1**2 + 0.5**2)**0.5), ("k_beta", 6.490, 0.17))
    LINES["pb"] = (("l_alpha2", 10.449, 0.11), ("l_alpha1", 10.551, 1), ("l_beta", 12.618, (0.5**2)**0.5))
    LINES["ti"] = (("k_alpha", 4.508, (1**2 + 0.5**2)**0.5), ("k_beta", 4.931, 0.15))
    LINES["zn"] = (("k_alpha", 8.637, (1**2 + 0.51**2)**0.5), ("k_beta", 9.572, 0.17))
    LINES["zr"] = (("k_alpha", 15.740, (1**2 + 0.51**2)**0.5), )
     
    NUM2SYMBOL = [symbol for symbol in LINES.keys()]
    SYMBOL2NUM = {symbol : i for i, symbol in enumerate(NUM2SYMBOL)}
    FANO = 5.8767709641972905
    CU_THRESHOLD_ENERGY = 8.037
    CU_ENERGY_INTENSITY_RATIO = 0.7
    ESC_THRESHOLD_ENERGY = 3
    ESCAPE_ENERGY_DIFF = 2.96
    DOUBLE_CU_ENERGY_DIFF = CU_THRESHOLD_ENERGY
    ESCAPE_ENERGY_INTENSITY_RATIO = 0.15

    @staticmethod
    def calculate_sigma(energy):
        return 0.51

    @staticmethod
    def get_parsed_element_lines(element):
        return [{"mu": line[1], "sigma": None, "intensity": line[2], "name": f"{element}_{line[0]}"} for line in Elements.LINES[element]]
    
    @staticmethod
    def get_element_line_energy(symbol, line_name):
        return [line[1] for line in Elements.LINES[symbol] if line[0] == line_name][0]