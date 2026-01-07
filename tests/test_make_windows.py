import pandas as pd
from src.data.make_windows import make_fixed_windows


def test_make_fixed_windows_shapes():
    df = pd.DataFrame({
        "f1": range(30),
        "f2": range(30),
        "label": [0]*15 + [1]*15
    })

    X, y = make_fixed_windows(
        df,
        ["f1", "f2"],
        "label",
        window_size=10,
        step=5
    )

    assert X.shape == (5, 10, 2)
    assert y.shape == (5,)
