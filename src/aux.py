from numpy import zeros, bincount, int8

def spike_count(indices, n):
    cout = zeros(n, dtype=int8)
    spike_out = bincount(indices)
    cout[:len(spike_out)] += spike_out.astype(int8)
    return cout
