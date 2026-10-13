# Arithmetic checks with a few hand-written numbers (no radar data generated).

import numpy as np

from metrics import classification_metrics, file_probabilities


def test_file_probabilities_average_each_file():

    probs = np.array([
        [0.9, 0.1, 0.0, 0.0],   # file 0
        [0.3, 0.7, 0.0, 0.0],   # file 0
        [0.0, 0.0, 1.0, 0.0],   # file 1
    ])

    result = file_probabilities(probs, [0, 0, 1], 2)

    assert np.allclose(result, [[0.6, 0.4, 0, 0], [0, 0, 1, 0]])


def test_metrics_values():

    m = classification_metrics([0, 0, 1, 1, 2, 3], [0, 1, 1, 1, 2, 3])

    assert m["accuracy"] == 5 / 6
    assert m["confusion_matrix"][0] == [1, 1, 0, 0]
    assert m["per_class"]["Inspire 2"]["recall"] == 0.5
    assert m["per_class"]["Matrice 30"]["precision"] == 2 / 3


def test_missing_class_still_four_by_four():

    m = classification_metrics([0, 0], [0, 1])

    assert np.array(m["confusion_matrix"]).shape == (4, 4)
