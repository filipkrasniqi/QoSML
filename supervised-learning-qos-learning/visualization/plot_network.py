import networkx as nx
from os import listdir, scandir
from os.path import isfile, join,expanduser, isdir
import numpy as np
import matplotlib.pyplot as plt

graph=nx.Graph()

arguments = ["plot_network.py", "abilene", "simulation_v1_L1", 0,0,0,0]
topology = arguments[1]
identifier = arguments[2]
intensity, simulation, capacity, prop_delay = int(arguments[3]),int(arguments[4]),int(arguments[5]),int(arguments[6])

base_dir = join(*[expanduser('~'), 'ns3', 'workspace', 'ns-allinone-3.29', 'ns-3.29', 'datasets', 'ns3'])
current_dir = join(*[base_dir, topology, identifier])+"/"

filename_links = join(*[current_dir, "intensity_{}_{}".format(intensity, simulation), "environment_{}_{}".format(capacity, prop_delay), "links.txt"])

links_f = open(filename_links, 'r')
all_nodes, all_links = [], []
# TODO list with pairs and unique nodes
for count_line, line in enumerate(links_f.readlines()):
    if count_line > 0:
        links_props = line.split(" ")
        pdelay_val_str = links_props[2]
        all_nodes.append(links_props[0])
        all_nodes.append(links_props[1])
        all_links.append((links_props[0], links_props[1]))
        all_links.append((links_props[1], links_props[0]))

all_nodes = np.unique(np.array(all_nodes))

graph.add_nodes_from(all_nodes)
graph.add_edges_from(all_links)
nx.draw(graph,with_labels=True)
plt.show()