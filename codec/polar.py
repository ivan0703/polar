import numpy as np
import math

class Polar:

    def __init__(self, N, K, ch_param, frozen):
        self.ch_type = 'bsc'
        self.N = N
        self.K = K
        self.n = math.ceil(math.log2(N))
        self.p = ch_param
        # TODO: determine frozen bits
        self.frozen = frozen

    # Construct frozenbits
    def construct(self, N, K, ch_param):
        return

    def encode(self, msg_bits):
        pass

    def decode(self, y):
        """
        Successive decoding
        :return:
        """
        pass



if __name__ == 'main':
    N = 8
    K = 4
    p = 0.1
    frozen = [0, 0, 0, -1, 0, -1, -1, -1]

    polar = Polar()
    polar.construct(N, K, p, frozen)