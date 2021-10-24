import torch
from torch_geometric.utils import to_undirected
import math


def cv_data_split(data, val_ratio: float = 0.16, test_ratio: float = 0.2):

    assert 'batch' not in data  # No batch-mode

    num_nodes = data.num_nodes
    row, col = data.edge_index # row = start nodes, col = end nodes
    data.edge_index = None

    mask = row < col
    row, col = row[mask], col[mask] # updated to be undirected now

    n_v = int(math.floor(val_ratio * row.size(0))) # number of validation edges (row.size(0) = len(row))
    n_t = int(math.floor(test_ratio * row.size(0))) # number of test edges

    
    # Positive edges -------------------------------------------------------------------------------------------
    perm_1 = torch.randperm(row.size(0))
    row, col = row[perm_1], col[perm_1]

    # ------------------- Fold 1 -------------------

    r, c = row[:n_v], col[:n_v] # 0 to n_v-1 entries (i.e. first n_v entries)
    data.val_pos_edge_index_1 = torch.stack([r, c], dim=0)


    r, c = row[len(row)-n_t:], col[len(row)-n_t:] # the last n_t entries
    data.test_pos_edge_index_1 = torch.stack([r, c], dim=0)


    r, c = row[n_v:len(row)-n_t], col[n_v:len(row)-n_t] # the remaining middle entries
    data.train_pos_edge_index_1 = torch.stack([r, c], dim=0)
    data.train_pos_edge_index_1 = to_undirected(data.train_pos_edge_index_1)
    

    # ------------------- Fold 2 -------------------

    r, c = row[n_v:2*n_v], col[n_v:2*n_v] # n_v to 2n_v-1 entries (i.e. second n_v entries)
    data.val_pos_edge_index_2 = torch.stack([r, c], dim=0)


    r, c = row[len(row)-n_t:], col[len(row)-n_t:] # the last n_t entries
    data.test_pos_edge_index_2 = torch.stack([r, c], dim=0)


    r, c = torch.cat((row[0:n_v], row[2*n_v:len(row)-n_t]), 0), torch.cat((col[0:n_v], col[2*n_v:len(row)-n_t]), 0) # the remaining entries
    data.train_pos_edge_index_2 = torch.stack([r, c], dim=0)
    data.train_pos_edge_index_2 = to_undirected(data.train_pos_edge_index_2)
    

    # ------------------- Fold 3 -------------------

    r, c = row[2*n_v:3*n_v], col[2*n_v:3*n_v] # 2n_v to 3n_v-1 entries (i.e. third n_v entries)
    data.val_pos_edge_index_3 = torch.stack([r, c], dim=0)


    r, c = row[len(row)-n_t:], col[len(row)-n_t:] # the last n_t entries
    data.test_pos_edge_index_3 = torch.stack([r, c], dim=0)


    r, c = torch.cat((row[0:2*n_v], row[3*n_v:len(row)-n_t]), 0), torch.cat((col[0:2*n_v], col[3*n_v:len(row)-n_t]), 0) # the remaining entries
    data.train_pos_edge_index_3 = torch.stack([r, c], dim=0)
    data.train_pos_edge_index_3 = to_undirected(data.train_pos_edge_index_3)
    

    # ------------------- Fold 4 -------------------

    r, c = row[3*n_v:4*n_v], col[3*n_v:4*n_v] # 3n_v to 4n_v-1 entries (i.e. fourth n_v entries)
    data.val_pos_edge_index_4 = torch.stack([r, c], dim=0)


    r, c = row[len(row)-n_t:], col[len(row)-n_t:] # the last n_t entries
    data.test_pos_edge_index_4 = torch.stack([r, c], dim=0)


    r, c = torch.cat((row[0:3*n_v], row[4*n_v:len(row)-n_t]), 0), torch.cat((col[0:3*n_v], col[4*n_v:len(row)-n_t]), 0) # the remaining entries
    data.train_pos_edge_index_4 = torch.stack([r, c], dim=0)
    data.train_pos_edge_index_4 = to_undirected(data.train_pos_edge_index_4)
    
    
    
    # ------------------- Fold 5 -------------------

    r, c = row[4*n_v:5*n_v], col[4*n_v:5*n_v] # 4n_v to 5n_v-1 entries (i.e. fith n_v entries)
    data.val_pos_edge_index_5 = torch.stack([r, c], dim=0)


    r, c = row[len(row)-n_t:], col[len(row)-n_t:] # the last n_t entries
    data.test_pos_edge_index_5 = torch.stack([r, c], dim=0)


    r, c = torch.cat((row[0:4*n_v], row[5*n_v:len(row)-n_t]), 0), torch.cat((col[0:4*n_v], col[5*n_v:len(row)-n_t]), 0)
    data.train_pos_edge_index_5 = torch.stack([r, c], dim=0)
    data.train_pos_edge_index_5 = to_undirected(data.train_pos_edge_index_5)
    
    
    
    # ------------------- Full -------------------

    r, c = row[len(row)-n_t:], col[len(row)-n_t:] # the last n_t entries
    data.test_pos_edge_index_f = torch.stack([r, c], dim=0)


    r, c = row[0:len(row)-n_t], col[0:len(row)-n_t] # the remaining entries
    data.train_pos_edge_index_f = torch.stack([r, c], dim=0)
    data.train_pos_edge_index_f = to_undirected(data.train_pos_edge_index_f)
    
    
    
    # Negative edges -------------------------------------------------------------------------------------------
    
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool) # nxn, upper triangular entries = True, rest = False
    neg_adj_mask[row, col] = 0 # adds False to wherever an edge exists (remaining Trues = negative edges)
    n1 = neg_adj_mask
    n2 = neg_adj_mask
    n3 = neg_adj_mask
    n4 = neg_adj_mask
    n5 = neg_adj_mask
    nf = neg_adj_mask
    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t() # returns undirected edges that dont exist (neg_row is start of non existent edge and neg_col is end)
    
    perm_2 = torch.randperm(neg_row.size(0))
    neg_row, neg_col = neg_row[perm_2], neg_col[perm_2]
    

    # ------------------- Fold 1 -------------------
    
    r, c = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index_1 = torch.stack([r, c], dim=0)

    r, c = neg_row[len(neg_row)-n_t:], neg_col[len(neg_row)-n_t:]
    data.test_neg_edge_index_1 = torch.stack([r, c], dim=0)
    
    n1[neg_row[:n_v], neg_col[:n_v]] = 0 # adds False to non existent edges being used for val instead
    n1[neg_row[len(neg_row)-n_t:], neg_col[len(neg_row)-n_t:]] = 0 # adds False to non existent edges being used for test instead
    data.train_neg_adj_mask_1 = n1

    
    # ------------------- Fold 2 -------------------
    
    
    r, c = neg_row[n_v:2*n_v], neg_col[n_v:2*n_v]
    data.val_neg_edge_index_2 = torch.stack([r, c], dim=0)

    r, c = neg_row[len(neg_row)-n_t:], neg_col[len(neg_row)-n_t:]
    data.test_neg_edge_index_2 = torch.stack([r, c], dim=0)
    
    n2[neg_row[n_v:2*n_v], neg_col[n_v:2*n_v]] = 0
    n2[neg_row[len(neg_row)-n_t:], neg_col[len(neg_row)-n_t:]] = 0
    data.train_neg_adj_mask_2 = n2
    
    
    # ------------------- Fold 3 -------------------
    
    
    r, c = neg_row[2*n_v:3*n_v], neg_col[2*n_v:3*n_v]
    data.val_neg_edge_index_3 = torch.stack([r, c], dim=0)

    r, c = neg_row[len(neg_row)-n_t:], neg_col[len(neg_row)-n_t:]
    data.test_neg_edge_index_3 = torch.stack([r, c], dim=0)
    
    n3[neg_row[2*n_v:3*n_v], neg_col[2*n_v:3*n_v]] = 0
    n3[neg_row[len(neg_row)-n_t:], neg_col[len(neg_row)-n_t:]] = 0
    data.train_neg_adj_mask_3 = n3
    
    
    # ------------------- Fold 4 -------------------
    
    
    r, c = neg_row[3*n_v:4*n_v], neg_col[3*n_v:4*n_v]
    data.val_neg_edge_index_4 = torch.stack([r, c], dim=0)

    r, c = neg_row[len(neg_row)-n_t:], neg_col[len(neg_row)-n_t:]
    data.test_neg_edge_index_4 = torch.stack([r, c], dim=0)
    
    n4[neg_row[3*n_v:4*n_v], neg_col[3*n_v:4*n_v]] = 0
    n4[neg_row[len(neg_row)-n_t:], neg_col[len(neg_row)-n_t:]] = 0
    data.train_neg_adj_mask_4 = n4
    
    
    # ------------------- Fold 5 -------------------
    
    
    r, c = neg_row[4*n_v:5*n_v], neg_col[4*n_v:5*n_v]
    data.val_neg_edge_index_5 = torch.stack([r, c], dim=0)

    r, c = neg_row[len(neg_row)-n_t:], neg_col[len(neg_row)-n_t:]
    data.test_neg_edge_index_5 = torch.stack([r, c], dim=0)
    
    n5[neg_row[4*n_v:5*n_v], neg_col[4*n_v:5*n_v]] = 0
    n5[neg_row[len(neg_row)-n_t:], neg_col[len(neg_row)-n_t:]] = 0
    data.train_neg_adj_mask_5 = n5
    
    
    # ------------------- Full -------------------
    

    r, c = neg_row[len(neg_row)-n_t:], neg_col[len(neg_row)-n_t:]
    data.test_neg_edge_index_f = torch.stack([r, c], dim=0)
    
    nf[neg_row[len(neg_row)-n_t:], neg_col[len(neg_row)-n_t:]] = 0
    data.train_neg_adj_mask_f = nf
    

    return data
