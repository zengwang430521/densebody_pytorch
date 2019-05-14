from torch_scatter import scatter_add


def batch_spmm(index, value, m, matrix):
    """Matrix product of sparse matrix with dense matrix in shape B * N * C.
    the sparse matrix is in size M * N, and there are L elements with non-zero values

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of sparse matrix.
        matrix (:class:`Tensor`): The dense matrix.

    :rtype: :class:`Tensor`
    """
    # denote len(value) as L
    row, col = index
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix[:, col, :]                              # B * L * C
    out = out * value.unsqueeze(-1).unsqueeze(0)         # B * L * C
    out = scatter_add(out, row, dim=1, dim_size=m)       # B * M * C

    return out

if __name__ == '__main__':
    import torch
    a = torch.rand(7, 8)
    t = a.to_sparse()
    index = t.indices()
    value = t.values()
    m = t.shape[0]
    b = torch.rand([3, 8, 4])
    c = torch.matmul(a, b)
    d = batch_spmm(index, value, m, b)
    print(c - d)