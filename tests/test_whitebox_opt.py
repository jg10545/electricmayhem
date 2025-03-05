from ax.service.ax_client import AxClient, ObjectiveProperties
import json
import os

from electricmayhem.whitebox._opt import _create_ax_client, _load_ax_client


def test_create_ax_client():
    params = {
        "foo":1,
        "bar":(1,2),
        "foobar":(1,10,"int"),
        "barfoo":(1,10, "log")
        }
    client = _create_ax_client("stuff", **params)
    assert isinstance(client, AxClient)
    
    

def test_create_ax_client_with_categoricals():
    params = {
        "foo":1,
        "bar":(1,2),
        "foobar":(1,10,"int"),
        "barfoo":(1,10, "log"),
        "optimizer":["bim", "mifgsm"],
        "lr_decay":"cosine"
        }
    
    client = _create_ax_client("stuff", **params)
    assert isinstance(client, AxClient)


def test_load_client(tmp_path_factory):
    logdir = str(tmp_path_factory.mktemp("images"))
    json_logpath = os.path.join(logdir, "log.json")
    # make a client
    params = {
        "foo":1,
        "bar":(1,2),
        "foobar":(1,10,"int"),
        "barfoo":(1,10, "log")
        }
    client = _create_ax_client("stuff", **params)

    # can't save without evaluating at least one trial
    def _evaluate_trial(p):
        return {"stuff":(1, 0.1)}
    N = 4
    for i in range(N):
        parameters, trial_index = client.get_next_trial()
        client.complete_trial(
                trial_index=trial_index, raw_data=_evaluate_trial(parameters))

    # save it to a JSON file
    j = client.to_json_snapshot()
    json.dump(j, open(json_logpath, "w"))
    # add some empty log directories
    for i in range(N):
        os.mkdir(os.path.join(logdir, f"{i}"))
    # now make sure we can read it back in
    newclient, startval = _load_ax_client(logdir)
    assert isinstance(newclient, AxClient)
    assert startval == N


