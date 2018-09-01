import numpy as np

__all__=['bsc', 'bsc_llr', 'bec', 'awgn']

def bsc(in_bits, p):
    """

    :param in_bits: Input bits
    :param p: Channel
    :return: Output bits
    """
    out_bits = in_bits.copy()
    flip_locs = (np.random.random(len(out_bits)) <= p)
    out_bits[flip_locs] = 1 ^ out_bits[flip_locs]
    return out_bits

def bsc_llr(in_bits, p):
    out_bits = bsc(in_bits, p)
    llr1 = np.log(p) - np.log(1 - p)    # LLR of y = 1, in BSC(p)
    out_llr = (2 * out_bits - 1) * llr1 # y is binary
    return out_llr


def bec():
    pass

def awgn():
    pass


if __name__ == '__main__':
    bsc_p = 0.1
    in_bits = np.array([0,0,1,0,1,1,0,0]);
    out_bits = bsc(in_bits, bsc_p)
    out_llr = bsc_llr(in_bits, bsc_p)

    print(f'Output buts = {out_bits}')
    print(f'Output llr = {out_llr}')