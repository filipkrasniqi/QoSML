import numpy as np
'''
Build columns for all datasets.
'''

def build_columns_only_load(num_nodes,start_send=0,end_send=-1, start_rec=0,end_rec=-1):
    if end_send < 0:
        end_send = num_nodes
    if end_rec < 0:
        end_rec = num_nodes
    columns = np.array([])
    for i in range(start_send,end_send):
        for j in range(start_rec, end_rec):
            columns = np.append(columns, 'load_{}_{}'.format(str(i),str(j)))
    return columns

def build_columns_only_delay_e2e(num_nodes,start_send=0,end_send=-1, start_rec=0,end_rec=-1):
    if end_send < 0:
        end_send = num_nodes
    if end_rec < 0:
        end_rec = num_nodes
    columns = np.array([])
    for i in range(start_send,end_send):
        for j in range(start_rec, end_rec):
            columns = np.append(columns, 'delay_e2e_{}_{}'.format(str(i),str(j)))
    return columns

def build_columns_only_delay_e2e_packet(num_nodes,start_send=0,end_send=-1, start_rec=0,end_rec=-1):
    if end_send < 0:
        end_send = num_nodes
    if end_rec < 0:
        end_rec = num_nodes
    columns = np.array([])
    for i in range(start_send,end_send):
        for j in range(start_rec, end_rec):
            columns = np.append(columns, 'delay_e2e_packet_{}_{}'.format(str(i),str(j)))
    return columns

def build_columns_only_delay_pl(num_nodes, start_send=0,end_send=-1, start_rec=0,end_rec=-1):
    if end_send < 0:
        end_send = num_nodes
    if end_rec < 0:
        end_rec = num_nodes
    columns = np.array([])
    for i in range(start_send,end_send):
        for j in range(start_rec, end_rec):
            columns = np.append(columns, 'delay_pl_{}_{}'.format(str(i),str(j)))
    return columns

def build_columns_only_traffic(num_nodes,start_send=0,end_send=-1, start_rec=0,end_rec=-1):
    if end_send < 0:
        end_send = num_nodes
    if end_rec < 0:
        end_rec = num_nodes
    columns = np.array([])
    for i in range(start_send,end_send):
        for j in range(start_rec, end_rec):
            columns = np.append(columns, 'traffic_{}_{}'.format(str(i),str(j)))
    return columns

def build_columns_only_dropped(num_nodes,start_send=0,end_send=-1, start_rec=0,end_rec=-1):
    if end_send < 0:
        end_send = num_nodes
    if end_rec < 0:
        end_rec = num_nodes
    columns = np.array([])
    for i in range(start_send,end_send):
        for j in range(start_rec, end_rec):
            columns = np.append(columns, 'dropped_{}_{}'.format(str(i),str(j)))
    return columns

def build_columns_routing(num_nodes):
    columns = []
    for i in range(0,num_nodes):
        for j in range(0, num_nodes):
            for k in range(0,num_nodes):
                for l in range(0, num_nodes):
                    columns.append('OD_{}_{} link_{}_{}'.format(str(i),str(j), str(k), str(l)))
    return columns

def build_columns_capacity(num_nodes):
    columns = []
    for i in range(0,num_nodes):
        for j in range(0, num_nodes):
            columns.append('capacity_{}_{}'.format(str(i),str(j)))
    return columns

def build_columns_only_links(num_nodes,start_send=0,end_send=-1, start_rec=0,end_rec=-1):
    if end_send < 0:
        end_send = num_nodes
    if end_rec < 0:
        end_rec = num_nodes
    columns = np.array([])
    for i in range(start_send,end_send):
        for j in range(start_rec, end_rec):
            columns = np.append(columns, 'capacity_{}_{}'.format(str(i),str(j)))
    for i in range(start_send,end_send):
        for j in range(start_rec, end_rec):
            columns = np.append(columns, 'queue_{}_{}'.format(str(i),str(j)))
    return columns
