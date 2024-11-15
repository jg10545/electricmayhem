import numpy as np
import torch
from tqdm import tqdm
import json
import os
import time
import kornia.geometry.transform

from ax.service.ax_client import AxClient, ObjectiveProperties
import ax.plot.diagnostic, ax.plot.scatter, ax.plot.contour
import ax.utils.notebook.plotting
import ax.modelbridge.cross_validation
import ax.modelbridge.generation_strategy
import ax.modelbridge.registry

from ._util import _bootstrap_std


def _create_ax_client(objective, minimize=True, **params):
    """
    :objective: string; name of metric to optimize on
    :minimize: bool; whether to maximize or minimize objective
    :params: parameters to
    """
    sampled_params = []

    for p in params:
        par = params[p]
        if isinstance(par, tuple):
            # add sampled param
            if len(par) == 2:
                sampled_params.append(
                    {
                        "name": p,
                        "type": "range",
                        "bounds": [par[0], par[1]],
                        "value_type": "float",
                    }
                )
            elif len(par) == 3:
                if par[-1] == "int":
                    sampled_params.append(
                        {
                            "name": p,
                            "type": "range",
                            "bounds": [par[0], par[1]],
                            "value_type": "int",
                        }
                    )
                elif par[-1] == "log":
                    sampled_params.append(
                        {
                            "name": p,
                            "type": "range",
                            "bounds": [par[0], par[1]],
                            "value_type": "float",
                            "log_scale": True,
                        }
                    )
                else:
                    assert False, f"I don't know what to do with {par[2]}"
            else:
                assert False, "I don't know what's even going on here"
        elif isinstance(par, list):
            sampled_params.append(
                {
                    "name": p,
                    "type": "choice",
                    "values": par,
                    "is_ordered": False,
                }
            )

        else:
            # add static param
            sampled_params.append({"name": p, "type": "fixed", "value": par})

    client = AxClient()
    client.create_experiment(
        name="pipeline_optimization",
        parameters=sampled_params,
        objectives={objective: ObjectiveProperties(minimize=minimize)},
    )
    return client
