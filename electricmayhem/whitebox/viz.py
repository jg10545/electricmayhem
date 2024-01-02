import numpy as np
import matplotlib.pyplot as plt



def eval_result_interactive(df):
    """
    Use ipywidgets to visualize eval results within a notebook
    """
    import ipywidgets
    def scatter(x, y, c):
        plt.cla()
        plt.scatter(df[x].values, df[y].values, 
                    c=df[c].values, alpha=0.5)
        plt.colorbar()
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(c)
    columns = list(df.columns)
    colors = [c for c in columns if c.endswith("_delta")|c.endswith("_patch")|c.endswith("_control")]
    ipywidgets.interact(scatter, x=columns, y=columns, c=colors)
    