import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def node_iter(G):
    return G.nodes()

def node_dict(G):
    node_dict = G.nodes
    return node_dict