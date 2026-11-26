import numpy as np
import pytest

from eval_traversability_npy import review_decisions


def test_agreement_counts_unknown_and_both_error_directions():
    labels = np.array([[1, 0, 1], [0, 0, np.nan]])
    decisions = np.array([[1, 1, 0], [np.nan, 0, 1]])
    assert review_decisions(decisions, labels).splitlines() == [
        "Agreement: 2/5 reviewed cells (40.0%)",
        "  human blocked, predicted traversable: 1",
        "  human traversable, predicted blocked: 1",
        "  unknown prediction on reviewed cells: 1",
    ]


@pytest.mark.parametrize("labels", [np.zeros((1, 2)), np.array([[2]]), np.array([[np.nan]])])
def test_invalid_labels_rejected(labels):
    with pytest.raises(ValueError):
        review_decisions(np.ones((1, 1)), labels)


def test_evaluate_course_export(tmp_path, monkeypatch, capsys):
    """Synthetic course with the measured Nov 6 heights and ramp angle."""
    import sys
    from eval_traversability_npy import main

    elevation = np.zeros((80, 100))
    elevation[10:30, 20:40] = 0.14
    elevation[40:60, 20:40] = 0.39
    elevation[:, 65:86] = np.arange(21) * 0.04 * np.tan(np.deg2rad(9))
    elevation[65, 50] = np.nan
    labels = np.full(elevation.shape, np.nan)
    labels[20, 10] = 1
    labels[20, 20] = labels[50, 20] = labels[35, 75] = 0
    np.save(tmp_path / "course.npy", elevation)
    np.save(tmp_path / "labels.npy", labels)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "eval_traversability_npy", "course.npy", "--resolution", "0.04",
        "--labels", "labels.npy", "--decisions-out", "decisions.npy",
    ])
    main()
    decisions = np.load("decisions.npy")
    reviewed = np.isfinite(labels)
    np.testing.assert_array_equal(decisions[reviewed], labels[reviewed])
    assert np.isnan(decisions[65, 50])
    assert "Agreement: 4/4 reviewed cells" in capsys.readouterr().out
