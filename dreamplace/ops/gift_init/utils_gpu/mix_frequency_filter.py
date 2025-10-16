import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse import (
    csc_matrix,
    identity,
)


def _sparsify_topk(adj_mat, k):
    """Keep at most k strongest outgoing edges per node."""
    if k is None or k <= 0:
        return adj_mat
    csr = adj_mat.tocsr()
    indptr = csr.indptr
    indices = csr.indices
    data = csr.data
    new_indptr = [0]
    new_indices = []
    new_data = []
    for row in range(csr.shape[0]):
        start = indptr[row]
        end = indptr[row + 1]
        row_indices = indices[start:end]
        row_data = data[start:end]
        if row_data.size == 0:
            new_indptr.append(len(new_indices))
            continue
        if row_data.size > k:
            topk_idx = np.argpartition(-np.abs(row_data), k - 1)[:k]
        else:
            topk_idx = np.arange(row_data.size)
        new_indices.extend(row_indices[topk_idx])
        new_data.extend(row_data[topk_idx])
        new_indptr.append(len(new_indices))
    sparsified = sp.csr_matrix(
        (np.array(new_data), np.array(new_indices), np.array(new_indptr)),
        shape=csr.shape,
        dtype=csr.dtype,
    )
    # symmetrize to maintain an undirected graph structure
    sparsified = 0.5 * (sparsified + sparsified.transpose())
    return sparsified.tocoo()


class GiFt_GPU(object):
    def __init__(self, adj_mat, device):
        self.adj_mat = adj_mat
        self.device = device 

    def train(self, sigma):

        adj_mat = self.adj_mat
        adj_mat = csc_matrix(adj_mat)
        dim = adj_mat.shape[0]
        start = time.time()
        adj_mat = adj_mat + sigma * identity(dim)  # augmented adj
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)  # D^-0.5
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj  # (D^-0.5(A+sigmaI)D^-0.5)
        self.d_mat = d_mat
        # self.norm_adj = norm_adj.tocsc()  # (D^-0.5(A+sigmaI)D^-0.5)
        # self.d_mat = d_mat.tocsc()
        end = time.time()
        # print('training time', end - start)

    def get_cell_position(self, k, cell_pos):
        norm_adj = self.norm_adj
        trainAdj = norm_adj.tocoo()
        edge_index = np.vstack((trainAdj.row, trainAdj.col)).transpose()
        edge_index = torch.from_numpy(edge_index).long()
        edge_index = edge_index.t().to(self.device)
        edge_weight = torch.from_numpy(trainAdj.data).float().to(self.device)

        norm_adj = torch.sparse.FloatTensor(edge_index,edge_weight).to(self.device)
        for _ in range(k):
            start = time.time()
            cell_pos = torch.sparse.mm(norm_adj, cell_pos)
            end = time.time()
        # torch.sparse.mm will create huge cache memory on GPU, which cannot be released automatically
        if cell_pos.is_cuda:
            torch.cuda.empty_cache()
        return cell_pos


class GiFtPlus_GPU(GiFt_GPU):
    def __init__(self, adj_mat, device, sparsify_k=None, fixed_mask=None):
        sparsified = _sparsify_topk(adj_mat, sparsify_k)
        super(GiFtPlus_GPU, self).__init__(sparsified, device)
        if fixed_mask is not None:
            self.fixed_mask_tensor = torch.from_numpy(fixed_mask).to(device)
        else:
            self.fixed_mask_tensor = None

    def get_cell_position_with_boundaries(
        self, k, cell_pos, fixed_locations, projection_steps=0
    ):
        norm_adj = self.norm_adj
        fixed_mask = self.fixed_mask_tensor
        for _ in range(k):
            cell_pos = torch.sparse.mm(norm_adj, cell_pos)
            if fixed_mask is not None:
                cell_pos[fixed_mask] = fixed_locations
        if projection_steps > 0:
            cell_pos = self.refine_with_boundaries(
                cell_pos, fixed_locations, projection_steps
            )
        if cell_pos.is_cuda:
            torch.cuda.empty_cache()
        return cell_pos

    def refine_with_boundaries(self, cell_pos, fixed_locations, iterations):
        if iterations <= 0:
            return cell_pos
        norm_adj = self.norm_adj
        fixed_mask = self.fixed_mask_tensor
        for _ in range(iterations):
            cell_pos = torch.sparse.mm(norm_adj, cell_pos)
            if fixed_mask is not None:
                cell_pos[fixed_mask] = fixed_locations
        if cell_pos.is_cuda:
            torch.cuda.empty_cache()
        return cell_pos
