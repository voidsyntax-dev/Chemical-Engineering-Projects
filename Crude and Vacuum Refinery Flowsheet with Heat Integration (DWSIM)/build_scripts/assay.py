"""Crude assay (user-supplied): Arab-Heavy type, 100 kbpd = 589,000 kg/h, SG 0.8891 (27.6 API wet),
2.85 wt% S, ~11 PTB salt, 0.25 wt% BS&W.  Real light ends + 22 fine TBP cuts (+ 11-component coarse merge)
with explicit MW / Tc / Pc / omega per cut.  Sulphur of the pseudo cuts is carried as a mass fraction (reporting).
"""
FEED_KG_H = 589000.0

# name -> wt % of whole crude (real compounds, DWSIM database names)
REAL_WT = {
    "Water": 0.2500, "Sodium Chloride": 0.0036, "Hydrogen sulfide": 0.0080,
    "Methane": 0.0200, "Ethane": 0.1200, "Propane": 0.3500, "Isobutane": 0.1200, "n-Butane": 0.2800,
    "Isopentane": 0.6000, "n-Pentane": 0.9000, "n-Hexane": 1.2000, "Methylcyclohexane": 1.1000,
    "Toluene": 0.9200, "n-Heptane": 1.1500, "n-Octane": 1.1500,
    "Methyl mercaptan": 0.0040, "Ethyl mercaptan": 0.0060, "Butane-1-Thiol": 0.0080, "Thiophene": 0.0120,
}
SG_REAL = {"Water": 1.0, "Sodium Chloride": 2.165, "Hydrogen sulfide": 0.79, "Methane": 0.42, "Ethane": 0.546,
           "Propane": 0.507, "Isobutane": 0.563, "n-Butane": 0.584, "Isopentane": 0.624, "n-Pentane": 0.631,
           "n-Hexane": 0.664, "Methylcyclohexane": 0.774, "Toluene": 0.872, "n-Heptane": 0.688, "n-Octane": 0.707,
           "Methyl mercaptan": 0.866, "Ethyl mercaptan": 0.839, "Butane-1-Thiol": 0.842, "Thiophene": 1.065}
S_FRAC_REAL = {"Hydrogen sulfide": 32.06 / 34.08, "Methyl mercaptan": 32.06 / 48.11, "Ethyl mercaptan": 32.06 / 62.13,
               "Butane-1-Thiol": 32.06 / 90.19, "Thiophene": 32.06 / 84.14}

# fine cuts: name, TBP C, SG, wt%, S wt%, MW, Tc K, Pc bar, omega
FINE = [
    ("K165", 165, 0.7680, 4.80, 0.35, 130, 640, 29.5, 0.45), ("K190", 190, 0.7830, 4.20, 0.45, 150, 675, 27.0, 0.55),
    ("K215", 215, 0.7978, 3.80, 0.55, 170, 690, 25.0, 0.64), ("K240", 240, 0.8122, 3.70, 0.65, 192, 715, 23.3, 0.73),
    ("D265", 265, 0.8262, 3.80, 0.90, 212, 745, 21.8, 0.81), ("D285", 285, 0.8374, 3.60, 1.10, 232, 765, 20.7, 0.88),
    ("D305", 305, 0.8483, 3.10, 1.35, 252, 775, 19.7, 0.94), ("D325", 325, 0.8590, 2.60, 1.60, 272, 790, 18.9, 1.00),
    ("A350", 350, 0.8721, 2.20, 2.10, 298, 820, 17.8, 1.06), ("A365", 365, 0.8799, 2.30, 2.35, 312, 832, 17.2, 1.09),
    ("V385", 385, 0.8901, 3.20, 2.20, 335, 848, 16.5, 1.12), ("V410", 410, 0.9026, 3.20, 2.45, 365, 870, 15.7, 1.16),
    ("V435", 435, 0.9149, 3.20, 2.70, 398, 890, 15.0, 1.20), ("V460", 460, 0.9270, 3.20, 2.95, 432, 910, 14.3, 1.23),
    ("V485", 485, 0.9388, 3.20, 3.20, 470, 930, 13.7, 1.26), ("V510", 510, 0.9505, 3.20, 3.45, 512, 948, 13.2, 1.28),
    ("V535", 535, 0.9620, 3.20, 3.70, 558, 965, 12.7, 1.30), ("V555", 555, 0.9711, 3.20, 3.90, 595, 978, 12.3, 1.31),
    ("R590", 590, 0.9867, 8.00, 4.50, 670, 1010, 11.7, 1.32), ("R640", 640, 1.0086, 9.00, 5.10, 770, 1045, 11.0, 1.32),
    ("R700", 700, 1.0341, 8.00, 5.70, 920, 1090, 10.3, 1.32), ("R780", 780, 1.0670, 7.0984, 6.07, 1220, 1150, 9.6, 1.32),
]

# coarse set used in the flowsheet: name, NBP C, SG, wt%, S wt%, MW, Tc K, Pc bar, omega
COARSE = [
    ("K178", 176.7, 0.7749, 9.00, 0.40, 139, 656, 28.3, 0.50), ("K228", 227.3, 0.8048, 7.50, 0.60, 181, 702, 24.2, 0.68),
    ("D275", 274.7, 0.8317, 7.40, 1.00, 222, 755, 21.3, 0.84), ("D315", 314.1, 0.8530, 5.70, 1.46, 261, 782, 19.3, 0.97),
    ("A357", 357.7, 0.8760, 4.50, 2.23, 305, 826, 17.5, 1.08), ("V397", 397.5, 0.8963, 6.40, 2.33, 350, 859, 16.1, 1.14),
    ("V447", 447.5, 0.9209, 6.40, 2.83, 415, 900, 14.7, 1.22), ("V497", 497.5, 0.9445, 6.40, 3.33, 491, 939, 13.5, 1.27),
    ("V545", 545.0, 0.9666, 6.40, 3.80, 577, 972, 12.5, 1.31), ("R615", 616.5, 0.9982, 17.00, 4.82, 723, 1029, 11.3, 1.32),
    ("R740", 737.6, 1.0494, 15.0984, 5.87, 1061, 1118, 10.0, 1.32),
]


def wt_fractions(coarse=True):
    """Mass fractions (sum = 1) for every component, plus the pseudo list used."""
    ps = COARSE if coarse else FINE
    w = {k: v / 100.0 for k, v in REAL_WT.items()}
    for p in ps:
        w[p[0]] = p[3] / 100.0
    return w, ps


def sulfur_fraction(name):
    if name in S_FRAC_REAL:
        return S_FRAC_REAL[name]
    for p in FINE + COARSE:
        if p[0] == name:
            return p[4] / 100.0
    return 0.0


def bulk(coarse=True):
    w, ps = wt_fractions(coarse)
    sg = dict(SG_REAL)
    sg.update({p[0]: p[2] for p in ps})
    vol = sum(w[k] / sg[k] for k in w)
    sg_wet = 1.0 / vol
    dry = {k: v for k, v in w.items() if k not in ("Water", "Sodium Chloride", "Hydrogen sulfide")}
    sg_dry = sum(dry.values()) / sum(v / sg[k] for k, v in dry.items())
    s_tot = 100 * sum(w[k] * sulfur_fraction(k) for k in w)
    return dict(sg_wet=sg_wet, api_wet=141.5 / sg_wet - 131.5, sg_dry=sg_dry, api_dry=141.5 / sg_dry - 131.5,
                sulfur_wt=s_tot, total_wt=100 * sum(w.values()))


if __name__ == "__main__":
    for c in (False, True):
        print("coarse" if c else "fine", bulk(c))
