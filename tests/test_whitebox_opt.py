from ax.service.ax_client import AxClient


from electricmayhem.whitebox._opt import _create_ax_client


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