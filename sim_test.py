import numpy as np
import math
import codec.polar as pc
import channels

N = 8
n = math.ceil(math.log2(N))
K = 4
p = 0.1

msg = np.array([1,0,0,1])

polar = pc.Polar(N, K, 'bsc', 0.1)
print(f'Polar({polar.N},{polar.K})')
print(f'bit reverse (0~7): {polar.bit_reversed_idx}')
print(f'frozen={polar.frozen}')

# encode
codeword = polar.encode(msg)
print(f'codeword={codeword}')
print(f'frozen={polar.frozen}')

out_bits = channels.bsc(codeword, p)
print(f'channel output bits = {out_bits}')

out_llr = channels.bsc_llr(codeword, p)
print(f'channel output llr = {out_llr}')


u = polar.decode(out_llr)
print(f'LLR={polar.LLR}')
print(f'u={u}')