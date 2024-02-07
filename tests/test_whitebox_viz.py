import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from electricmayhem.whitebox import viz


def test_eval_result_permutation_importance():
    N = 100
    df = {"mymetric_patch":np.random.normal(0,1,N),
          "categorical_col":np.random.choice(["foo", "bar"], size=N)}
    for col in ["foo", "bar", "foobar", "barfoo"]:
        df[col] = np.random.normal(0,1,100)
    df = pd.DataFrame(df)
    
    plot = viz.eval_result_permutation_importance(df,"mymetric_patch", 
                                                  n_estimators=3,
                                                  n_repeats=2)
    assert isinstance(plot, plt.Axes)