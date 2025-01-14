import math

import torch
from torch_geometric.utils import to_undirected  # type: ignore


def cv_data_split(data, val_ratio: float = 0.16, test_ratio: float = 0.2):
    assert "batch" not in data  # no batch-mode

    num_nodes = data.num_nodes
    row, col = data.edge_index
    data.edge_index = None

    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # positive edges
    perm_1 = torch.randperm(row.size(0))
    row, col = row[perm_1], col[perm_1]

    for fold in range(1, 6):
        start_v = (fold - 1) * n_v
        end_v = fold * n_v
        val_indices = slice(start_v, end_v)
        train_indices = torch.cat((row[:start_v], row[end_v : len(row) - n_t])), torch.cat(
            (col[:start_v], col[end_v : len(row) - n_t])
        )

        setattr(data, f"val_pos_edge_index_{fold}", torch.stack([row[val_indices], col[val_indices]], dim=0))
        setattr(data, f"test_pos_edge_index_{fold}", torch.stack([row[len(row) - n_t :], col[len(row) - n_t :]], dim=0))
        train_pos_edge_index = torch.stack(train_indices, dim=0)
        setattr(data, f"train_pos_edge_index_{fold}", to_undirected(train_pos_edge_index))

    # full dataset
    data.test_pos_edge_index_f = torch.stack([row[len(row) - n_t :], col[len(row) - n_t :]], dim=0)
    data.train_pos_edge_index_f = to_undirected(torch.stack([row[: len(row) - n_t], col[: len(row) - n_t]], dim=0))

    # negative edges
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8).triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0
    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()

    perm_2 = torch.randperm(neg_row.size(0))
    neg_row, neg_col = neg_row[perm_2], neg_col[perm_2]

    for fold in range(1, 6):
        start_v = (fold - 1) * n_v
        end_v = fold * n_v
        val_indices = slice(start_v, end_v)

        setattr(data, f"val_neg_edge_index_{fold}", torch.stack([neg_row[val_indices], neg_col[val_indices]], dim=0))
        setattr(
            data,
            f"test_neg_edge_index_{fold}",
            torch.stack([neg_row[len(neg_row) - n_t :], neg_col[len(neg_row) - n_t :]], dim=0),
        )

        neg_adj_mask_copy = neg_adj_mask.clone()
        neg_adj_mask_copy[neg_row[val_indices], neg_col[val_indices]] = 0
        neg_adj_mask_copy[neg_row[len(neg_row) - n_t :], neg_col[len(neg_row) - n_t :]] = 0
        setattr(data, f"train_neg_adj_mask_{fold}", neg_adj_mask_copy)

    # full dataset
    data.test_neg_edge_index_f = torch.stack([neg_row[len(neg_row) - n_t :], neg_col[len(neg_row) - n_t :]], dim=0)
    neg_adj_mask[len(neg_row) - n_t :, len(neg_row) - n_t :] = 0
    data.train_neg_adj_mask_f = neg_adj_mask

    return data
