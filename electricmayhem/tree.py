import numpy as np
import pandas as pd

def notebook_view_pydot(pdot):
    """
    
    """
    from IPython.display import Image, display
    plt = Image(pdot.create_png())
    display(plt)
    


def _mincost(df, c):
    """
    Traverse tree to get minimum cost for each node
    """
    # for convenience, build a dictionary of children for
    # each parent and AND/OR type for each node
    childdict = {}
    for n in df.name.unique():
        childdict[n] = list(df[df.parent == n].name.values)
    
    typedict = {}
    for i in range(len(df)):
        typedict[df.name.values[i]] = df["or"].values[i]
        
    mincosts = {}
    mincostnodes = {}

    for e, d in df[pd.notnull(df.cost)].iterrows():
        mincosts[d["name"]] = d["cost"]
        mincostnodes[d["name"]] = [d["name"]]
    
    unprocessed = list(df[pd.isnull(df.cost)].name.values)
    # as long as there are unprocessed nodes
    while len(unprocessed) > 0:
        # pick one and see if all the kids are ready
        i = np.random.randint(0, len(unprocessed))
        n = unprocessed[i]
        all_children_computed = True
        for c in childdict[n]:
            if c not in mincosts:
                all_children_computed = False
            
        if all_children_computed:
            # OR node
            if typedict[n]:
                cost = np.inf
                # go through each kid- if the cost goes down
                # then keep it
                for k in childdict[n]:
                    if mincosts[k] < cost:
                        cost = mincosts[k]
                        path = mincostnodes[k] + [n]
            # AND node
            else:
                # go through each kid and add them up
                path = [n]
                cost = 0
                for k in childdict[n]:
                    cost += mincosts[k]
                    path += mincostnodes[k]
                
            mincosts[n] = cost
            mincostnodes[n] = path
        
            # remove from queue
            _ = unprocessed.pop(i)
    return mincosts, mincostnodes


def build_tree(data, name="attack tree", sublabel=None, mincost=None):
    """
    Build a PyDot object representing an attack tree.
    
    Input dataframe should have the following labels:
        -name: unique descriptor for node
        -label: actual label you want displayed
        -or: True for OR nodes; False for AND
        -parent: name of parent node; None for root
        
    as well as any other columns containing useful information
    
    :data: pandas DataFrame containing node info
    :name: string; name of PyDot graph object
    :sublabel: string; give name of a column to append to display 
        labels
    :mincost: string; display minimum cost subtree relative to this 
        column. Assumes that ONLY basic attack steps (leaf nodes) have
        this value filled in
        
    Returns PyDot object. OR nodes will be drawn as houses; AND nodes
    will be squares, and basic attack steps (leaf nodes) will be 
    ellipses.
    """
    import pydot
    parents = data.parent.unique()
    graph = pydot.Dot(name, 
                  graph_type="digraph")
    
    if mincost is not None:
        mincosts, mincostnodes = _mincost(data, mincost)
        # assume a single root
        root = data[pd.isnull(data.parent)].name.values[0]
        mincostpath = mincostnodes[root]
        
    # ADD NODES
    for i,d in data.iterrows():
        if d["name"] not in parents:
            shape = "ellipse"
        elif d["or"]:
            shape = "house"
        else:
            shape = "rectangle"
        label = d["label"]
        color = "black"
        if sublabel is not None:
            label += "\n" + str(d[sublabel])
        if mincost is not None:
            label += "\n" + str(mincosts[d["name"]])
            if d["name"] not in mincostpath:
                color = "gray"
        node = pydot.Node(d["name"], label=label, shape=shape, color=color)
        graph.add_node(node)
    
    # ADD EDGES
    for i,d in data.iterrows():
        if d["parent"] is not None:
            if mincost is not None:
                if (d["parent"] in mincostpath)&(d["name"] in mincostpath):
                    style = "solid"
                    color = "black"
                else:
                    style = "dashed"
                    color = "gray"
            else:
                style = "solid"
                color = "black"
            edge = pydot.Edge(d["parent"], d["name"], style=style,
                             color=color)
            graph.add_edge(edge)
    return graph