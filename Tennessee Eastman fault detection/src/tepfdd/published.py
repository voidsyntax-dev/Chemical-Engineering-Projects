"""Published per-fault detection rates (%), transcribed by hand from Agarwal, Gonzalez, Elkamel & Budman,
"Hierarchical Deep Recurrent Neural Network based Method for Fault Detection and Diagnosis", arXiv:2012.03861 (preprint),
Tables 4 and 5. That paper itself quotes the PCA / SAE-NN / DSN / GAN / OCSVM / CNN numbers from earlier studies, so they are
SECOND-HAND. Their data set and protocol are not identical to the Rieth et al. official split used here, and they do not report
a false-alarm-matched detection rate, so these are context, not a like-for-like benchmark.
Table 4 (non-incipient faults only): PCA with 15 components, T^2 and SPE.   Table 5 (all faults): the remaining methods.
Faults 3, 9, 15 are absent from Table 4 (PCA), hence None.
"""
SOURCE = ("Agarwal et al., arXiv:2012.03861, Tables 4-5 (values for PCA, SAE-NN, DSN, GAN, OCSVM, CNN are quoted by that paper "
          "from earlier work; HDRNN = the paper's own LSTM-autoencoder hierarchy)")
METHODS = ["PCA-T2", "PCA-SPE", "SAE-NN", "DSN", "GAN", "OCSVM", "CNN", "HDRNN"]
# fault: (PCA T2, PCA SPE, SAE-NN, DSN, GAN, OCSVM, CNN, HDRNN)
TABLE = {
    1: (99.2, 99.8, 77.6, 90.8, 99.62, 99.5, 91.39, 100), 2: (98, 98.6, 85, 89.6, 98.5, 98.5, 87.96, 100),
    3: (None, None, 79.4, 14.4, 10.375, 7.62, 50.59, 81.58), 4: (4.4, 96.2, 56.6, 47.6, 56.25, 50.37, 99.73, 100),
    5: (22.5, 25.4, 76, 31.6, 32.37, 30.5, 90.35, 100), 6: (98.9, 100, 82.8, 91.6, 100, 100, 91.5, 100),
    7: (91.5, 100, 80.6, 91, 99.99, 99.62, 91.55, 100), 8: (96.6, 97.6, 83, 90.2, 97.87, 97.37, 82.95, 100),
    9: (None, None, 50.6, 16.3, 8.625, 7.125, 49.53, 99.38), 10: (33.4, 34.1, 75.3, 63.2, 50.87, 53.25, 70.05, 42.84),
    11: (20.6, 64.4, 75.9, 54.2, 58, 54.75, 60.16, 100), 12: (97.1, 97.5, 83.3, 87.8, 98.75, 98.63, 85.56, 100),
    13: (94, 95.5, 83.3, 85.5, 95, 94.87, 46.92, 100), 14: (84.2, 100, 77.8, 89, 100, 100, 88.88, 100),
    15: (None, None, 55.5, 26.7, 12.5, 14, 43.54, 100), 16: (16.6, 24.5, 78.3, 74.8, 34.37, 36.37, 66.84, 100),
    17: (74.1, 89.2, 78, 83.3, 91.12, 87.25, 77.11, 100), 18: (88.7, 89.9, 83.3, 82.4, 90.37, 90.12, 82.74, 100),
    19: (0.4, 12.7, 67.7, 52.4, 11.8, 3.75, 70.87, 40.4), 20: (29.9, 45, 77.1, 44.1, 58.37, 52.75, 72.88, 100),
}
AVERAGE_ALL_FAULTS = {"SAE-NN": 75.355, "DSN": 65.32, "GAN": 64.51, "OCSVM": 62.78, "CNN": 79.84, "HDRNN": 93.23}
AVERAGE_NON_INCIPIENT_PCA = {"PCA-T2": 61.77, "PCA-SPE": 74.72}   # 17 faults (without 3, 9, 15), Table 4
