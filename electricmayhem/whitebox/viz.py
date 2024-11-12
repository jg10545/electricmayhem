import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection, sklearn.ensemble, sklearn.inspection


def eval_result_interactive(df):
    """
    Use ipywidgets to visualize eval results within a notebook
    """
    import ipywidgets

    def scatter(x, y, c):
        plt.cla()
        plt.scatter(df[x].values, df[y].values, c=df[c].values, alpha=0.5)
        plt.colorbar()
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(c)

    columns = list(df.columns)
    colors = [
        c
        for c in columns
        if c.endswith("_delta") | c.endswith("_patch") | c.endswith("_control")
    ]
    ipywidgets.interact(scatter, x=columns, y=columns, c=colors)


def eval_result_permutation_importance(
    df,
    metric,
    n_estimators=100,
    test_size=0.3,
    n_repeats=25,
    max_features_to_plot=25,
    n_jobs=-1,
):
    """
    Plotting macro to try and discover which pipeline parameters are determining variability in
    patch performance. The basic idea is to fit a random forest regressor to the eval results
    and then do a permutation importance plot on the forest.

    If you have categorical columns that are strings it should identify them and break out into
    dummy variables. If you have categorical columns that are integers it will probably do
    something stupid.

    Permutation plot follows code on the scikit-learn docs:
        https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py

    :df: pandas DataFrame containing eval results
    :metric: string; column name for the metric to fit against
    :n_estimators: int; number of trees in RandomForestRegressor
    :test_size: float; fraction of df to use for test set
    :n_repeats: int; number of permutations to test
    :max_features_to_plot: int; maximum number of features to show
    :n_jobs: int; n_jobs argument to pass to joblib
    """
    # FIRST FIGURE OUT WHICH COLUMNS ARE WHICH
    metric_names = [
        c
        for c in df.columns
        if c.endswith("_patch") | c.endswith("_delta") | c.endswith("_control")
    ]
    feature_names = [c for c in df.columns if c not in metric_names]
    categoricals = [c for c in feature_names if df[c].dtype == "O"]
    noncategoricals = [c for c in feature_names if c not in categoricals]

    # make a new dataframe with the categoricals broken out to dummy variables
    newdf = df[noncategoricals]
    if len(categoricals) > 0:
        for c in categoricals:
            newdf = pd.concat(
                [newdf, pd.get_dummies(df[c], prefix=c, dtype="int")], axis=1
            )

    # break into training and test splits
    X = newdf.values
    Y = df[metric].values
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=test_size
    )

    # initialize a random forest regressor and train it
    forest = sklearn.ensemble.RandomForestRegressor(n_estimators, n_jobs=n_jobs)
    forest.fit(X_train, Y_train)

    # now run the permutation test. this step can take a while, especially if there are
    # a lot of trees in the forest
    perm_result = sklearn.inspection.permutation_importance(
        forest, X_test, Y_test, n_repeats=n_repeats, n_jobs=n_jobs
    )

    # now plot it
    sorted_importances_idx = perm_result.importances_mean.argsort()
    importances = pd.DataFrame(
        perm_result.importances[sorted_importances_idx].T,
        columns=newdf.columns[sorted_importances_idx],
    )
    importances = importances[importances.columns[-max_features_to_plot:]]

    ax = importances.plot.box(vert=False, whis=10)
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_title(f"Permutation Importance for {metric}")
    ax.set_xlabel("Decrease in squared error")
    return ax
