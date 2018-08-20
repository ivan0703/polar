import numpy as np
import math

class Polar:

    @staticmethod
    def reverse(x, n):
        """

        :rtype: object
        """
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

        # TODO: check llr_y == self.N

        u = np.zeros(self.N)
        d_hat = np.zeros(self.N)
        self.LLR[0:self.N] = 0
        self.LLR[self.N-1:] = llr_y

        for j in range(self.N):
            i = Polar.reverse(j, self.n)

            # Step 1 update LLR
            self._updateLLR(i)

            # Step 2 update d_hat
            if frozen[i] == -1:
                if self.LLR[0] > 0:
                    d_hat[i] = 0
                else:
                    d_hat[i] = 1
            else:
                d_hat[i]

            # Step 3 update BITS
            self._updateBITS(d_hat[i], i)



        return u

    def _updateLLR(self, idx):
        pass

    def _updateBITS(self, latest_bit, idx):
        pass


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
