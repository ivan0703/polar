import numpy as np
import math
#import channels

class Polar:

    def __init__(self, N, K, channel_type, channel_param):
        """

        :param N: Block length
        :param K: Message length
        :param channel_type: 'bsc', 'awgn'
        :param channel_param:
        """

        self.N = N
        self.K = K
        self.channel_type = channel_type
        self.channel_param = channel_param

        self.n = math.ceil(math.log2(N))
        self.LLR = np.zeros(2*N - 1)
        self.BITS = np.zeros((2, N - 1))
        self.bit_reversed_idx = np.zeros(N)
        self.frozen = self._construct()

        for i in range(N):
            self.bit_reversed_idx[i] = reverse(i, self.n)



    def _construct(self):
        """

        :return: frozen
        """
        # log domain z-value
        z = np.zeros(self.N)

        # initialize
        frozen = np.full(self.N, -1)

        if(self.channel_type == 'bsc'):
            z[0] = np.log(2) + 0.5 * np.log(self.channel_param) + 0.5 * np.log(1-self.channel_param)
        elif(self.channel_type == 'awgn'):
            pass

        for lev in range(self.n):
            # number of node at lev
            lev_node = np.power(2, lev)

            # Use z-values on lev to generate z-values on lev+1
            for i in range(lev_node):
                T = z[i]
                z[i] = log_diff(np.log(2)+T, 2*T) # 2z - z^2 in log domain
                z[i+lev_node] = 2*T               # z^2 in log domain

        # Sorting the resulting z-values
        sorted_idx = np.argsort(z)

        # The last N-K indices in sorted_idx is to be frozen
        frozen[np.sort(sorted_idx[self.K:])] = 0

        return frozen



    def encode(self, u):
        """

        :param u: message bits
        :return: coded bits
        """

        x = self.frozen.copy()
        x[x==-1] = u    # len(u) = K

        # loop through level
        for i in range(self.n):
            B  = np.power(2, self.n-i)
            nB = np.power(2,i)
            halfB = int(B/2)

            # loop through nB
            for n in range(nB):
                base = n*B
                for j in range(halfB):
                    x[base+j] = np.mod(x[base+j] + x[base+j+halfB], 2)

        return x

    def decode(self, llr_y):
        """
        Successive decoding
        :return:
        """
        # TODO: check llr_y == self.N
        u = np.zeros(self.N)
        d_hat = np.zeros(self.N)
        self.LLR[0:self.N] = 0
        self.LLR[self.N-1:] = llr_y

        # print(f'LLR={self.LLR}')
        print(f'frozen={self.frozen}')

        for j in range(self.N):
            i = reverse(j, self.n)

            # Step 1 update LLR
            self._updateLLR(i)

            # print(f'>>> Step {j}')
            # print(f'>>>> LLR:')
            # print(f'{self.LLR}')
            # print(f'>>>> BITS:')
            # print(f'{self.BITS}')

            # Step 2 update d_hat
            if self.frozen[i] == -1:
                if self.LLR[0] > 0:
                    d_hat[i] = 0
                else:
                    d_hat[i] = 1
            else:
                d_hat[i] = self.frozen[i]

            # Step 3 update BITS
            self._updateBITS(d_hat[i], i)

        print(f'd_hat={d_hat}')
        print(f'frozen={self.frozen}')

        # return codeword
        return d_hat[self.frozen == -1]

    def _updateLLR(self, idx):
        nextlevel = self.n
        if idx > 0:
            # Update lastlevel using lowerconv
            lastlevel = first1_index(idx, self.n)
            # Indices of the lastlevel LLR in self.LLR
            start = np.power(2,lastlevel-1) - 1
            end   = np.power(2,lastlevel) - 1       # not included
            for i in range(start, end):
                self.LLR[i] = lowerconv(self.BITS[0][i], self.LLR[2*(i+1)-1], self.LLR[2*(i+1)])
            nextlevel = lastlevel - 1

        # Update LLR at nextlevel down to 1
        for lev in range(nextlevel, 0, -1):
            # Indices of the lev LLR in self.LLR
            start = np.power(2,lev-1) - 1
            end = np.power(2,lev) - 1       # not included
            for i in range(start, end):
                self.LLR[i] = upperconv(self.LLR[2*(i+1)-1], self.LLR[2*(i+1)])

    def _updateBITS(self, latest_bit, idx):
        if idx == self.N - 1:
            return
        elif idx < self.N / 2:
            self.BITS[0][0] = latest_bit
        else:
            lastlevel = first0_index(idx, self.n)

            # Update BITS[1][0]; level 1
            self.BITS[1][0] = latest_bit

            # Update BITS[1][1 ~ lastlevel-2]; bits on 2 ~ lastlevel-1 levels
            for lev in range(2, lastlevel):
                # using (lev-1) bits to update lev bits
                known_bit_lev = lev - 1
                start = np.power(2,known_bit_lev-1) - 1
                end = np.power(2,known_bit_lev) - 1   # not included
                for i in range(start, end):
                    self.BITS[1][2*(i+1)-1] = np.mod(self.BITS[0][i] + self.BITS[1][i], 2)
                    self.BITS[1][2*(i+1)]   = self.BITS[1][i]

            # Update BITS[0][lastlevel-1]; bits on lastlevel level
            known_bit_lev = lastlevel - 1
            start = np.power(2, known_bit_lev - 1) - 1
            end = np.power(2, known_bit_lev) - 1  # not included
            for i in range(start, end):
                self.BITS[0][2 *(i+1)-1] = np.mod(self.BITS[0][i] + self.BITS[1][i], 2)
                self.BITS[0][2 *(i+1)]   = self.BITS[1][i]

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
        return l2 + np.log1p(np.exp(l1-l2))
    else:
        return l1 + np.log1p(np.exp(l2-l1))

def log_diff(l1, l2):
    """
    Calculate log( exp(l1) - exp(l2) ); l1 > l2 is required
    """
    return l1 + np.log1p(-np.exp(l2-l1))

def upperconv(llr1, llr2):
    return log_sum(llr1+llr2, 0) - log_sum(llr1, llr2)

if __name__ == '__main__':
    N = 8
    n = math.ceil(math.log2(N))
    K = 4
    p = 0.1

    msg = np.array([1,0,0,1])

    polar = Polar(N, K, 'bsc', 0.1)
    print(f'Polar({polar.N},{polar.K})')
    print(f'bit reverse (0~7): {polar.bit_reversed_idx}')
    print(f'frozen={polar.frozen}')

    # encode
    codeword = polar.encode(msg)
    print(f'codeword={codeword}')
    print(f'frozen={polar.frozen}')

    # out_bits = channels.bsc(codeword, p)
    # print(f'channel output bits = {out_bits}')

    # out_llr = channels.bsc_llr(codeword, p)
    # print(f'channel output llr = {out_llr}')


    # u = polar.decode(out_llr)
    # print(f'LLR={polar.LLR}')
    # print(f'u={u}')