import numpy as np

__all__=['bsc', 'bec', 'awgn']

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

def bec():
    pass

def awgn():
    pass


if __name__ == '__main__':
    bsc_p = 0.1
    in_bits = np.array([0,0,1,0,1,1,0,0]);
    out_bits = bsc(in_bits,bsc_p)

    print(f'Output buts = {out_bits}')