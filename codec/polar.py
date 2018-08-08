import numpy as np
import math

class Polar:

    @staticmethod
    def reverse(x, n):
        result = 0
        for i in range(n):
            if (x >> i) & 1: result |= 1 << (n - 1 - i)
        return result

    def __init__(self, N, K):
        self.ch_type = 'bsc'
        self.N = N
        self.K = K
        self.n = math.ceil(math.log2(N))
        self.LLR = np.zeros(2*N - 1)
        self.BITS = np.zeros((2, N - 1))
        self.bit_reversed_idx = np.zeros(N)

        for i in range(N):
            self.bit_reversed_idx[i] = self.reverse(i, self.n)



    # Construct frozenbits
    def construct(self, N, K, channel):
        return

    def encode(self, msg_bits):
        pass

    def decode(self, llr_y, frozen):
        """
        Successive decoding
        :return:
        """
        u = np.zeros(len(llr_y))
        self.LLR[0:N] = 0
        self.LLR[self.N-1:] = llr_y

        return u




if __name__ == '__main__':
    N = 8
    K = 4
    p = 0.1
    frozen = [0, 0, 0, -1, 0, -1, -1, -1]
    llr_y = [1, 0, 1, 0, 0, 1, 0, 0]

    polar = Polar(N, K)
    print(f'Polar({polar.N},{polar.K})')
    print(f'bit reverse (0~7): {polar.bit_reversed_idx}')

    u = polar.decode(llr_y, frozen)
    print(f'{polar.LLR}')
    print(f'u={u}')
