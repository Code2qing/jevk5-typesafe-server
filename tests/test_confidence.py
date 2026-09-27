from jevk5_server.confidence import calculate_confidence


def test_confidence_all_on_one_option():
    # When all probability is on one option, confidence is 1.0
    assert calculate_confidence([1.0, 0.0, 0.0]) == 1.0


def test_confidence_uniform_distribution():
    # When probabilities are evenly spread, confidence is 0.0
    assert calculate_confidence([1 / 3, 1 / 3, 1 / 3]) == 0.0


def test_confidence_single_option():
    # With a single option, confidence is 1.0
    assert calculate_confidence([1.0]) == 1.0


def test_confidence_example_from_docs():
    # From TypeSafe doc: n=3, peak=0.95 -> (3 * 0.95 - 1) / 2 = 0.925 -> approx 0.92
    conf = calculate_confidence([0.0, 0.95, 0.05])
    assert round(conf, 2) == 0.92
