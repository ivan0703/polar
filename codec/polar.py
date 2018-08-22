import numpy as np

class Polar:

    def __init__(self, N, K):
        self.ch_type = 'bsc'
        self.N = N
        self.K = K
        self.n = np.ceil(np.log2(N))
        self.LLR = np.zeros(2*N - 1)
        self.BITS = np.zeros((2, N - 1))
        self.bit_reversed_idx = np.zeros(N)

        for i in range(N):
            self.bit_reversed_idx[i] = reverse(i, self.n)



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
            i = reverse(j, self.n)

            # Step 1 update LLR
            self._updateLLR(i)

            # Step 2 update d_hat
            if frozen[i] == -1:
                if self.LLR[0] > 0:
                    d_hat[i] = 0
                else:
                    d_hat[i] = 1
            else:
                d_hat[i] = frozen[i]

            # Step 3 update BITS
            self._updateBITS(d_hat[i], i)



        return u

    def _updateLLR(self, idx):
        nextlevel = self.n
        if idx > 0:
            # Update lastlevel using lowerconv
            lastlevel = first1_index(idx, self.n)
            # Indices of the lastlevel LLR in self.LLR
            start = np.power(2,lastlevel-1) - 1
            end   = np.power(2,lastlevel) - 1       # not included
            for i in range(start, end):
                self.LLR = lowerconv(self.BITS(i), self.LLR(2*(i+1)), self.LLR(2*(i+1)+1))
            nextlevel = lastlevel - 1

        # Update LLR at nextlevel down to 1
        for lev in range(nextlevel, 0, -1):
            # Indices of the lev LLR in self.LLR
            start = np.power(2,lev-1) - 1
            end = np.power(2,lev) - 1       # not included
            for i in range(start, end):
                self.LLR = upperconv(self.LLR(2*(i+1)), self.LLR(2*(i+1)+1))

    def _updateBITS(self, latest_bit, idx):
        if idx == self.N - 1:
            return
        elif idx < self.N / 2:
            self.BITS[0][0] = latest_bit
        else:
            lastlevel = first0_index(idx, self.n)
            self.BITS[1][0] = latest_bit



def reverse(x, n):
    """

    :rtype: object
    """
    result = 0
    for i in range(n):
        if (x >> i) & 1: result |= 1 << (n - 1 - i)
    return result


def first1_index(num, n):
    result = n
    for i in range(n-1):
        if((num & (1<<(n-1-i))) != 0):
            result = i + 1
            break
    return result

def first0_index(num, n):
    result = n
    for i in range(n-1):
        if((num & (1<<(n-1-i))) == 0):
            result = i + 1
            break
    return result

def lowerconv(upper_bit, upper_llr, lower_llr):
    if upper_bit == 0:
        return lower_llr + upper_llr
    else:
        return lower_llr - upper_llr

def log_sum(l1, l2):
    """
    Calculate log( exp(l1) + exp(l2) )
    """
    if(l1 < l2):
        return l2 + np.log1p(l1-l2)
    else:
        return l1 + np.log1p(l2-l1)

def upperconv(llr1, llr2):
    return log_sum(llr1+llr2, 0) - log_sum(llr1, llr2)

if __name__ == '__main__':
    N = 8
    n = np.ceil(np.log2(N))
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

    # test first1_index
    for num in range(N):
        print(f'first1({num},{n})={first1_index(num,n)}')

    # test first0_index
    for num in range(N):
        print(f'first0({num},{n})={first0_index(num,n)}')