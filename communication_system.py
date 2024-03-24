# 导包
from commpy.channelcoding import cyclic_code_genpoly
from commpy.utilities import decimal2bitarray as dec2bit
from commpy.utilities import bitarray2dec as bit2dec
from commpy.utilities import upsample

from commpy.filters import rcosfilter
import math
import numpy as np

from scipy import interpolate as interp
from scipy.signal import firwin

# generate random binary sequence set


def gen_rand_code(code_len, sample_num):
    return np.random.randint(0, 2, [sample_num, code_len], dtype=np.uint8)


# channel coding

class Hamming74:
    _k = 4
    _n = 7

    def __init__(self):
        # 生成矩阵部分
        self._P = np.array([
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 1, 1]
        ])

        # 监督矩阵
        self._H = np.concatenate([self._P, np.eye(3, dtype=int)], axis=1)

        self._C = np.zeros([8, 4])
        err_bit = [-1, -1, -1, 3, -1, 2, 1, 0]
        for n, err in enumerate(err_bit):
            if err >= 0:
                self._C[n, err] = 1

    def encode(self, X):
        code_num = len(X)
        X_unfold = X.reshape([-1, self._k])
        A_unfold = (X_unfold @ self._P.T) % 2
        Y_unfold = np.concatenate([X_unfold, A_unfold], axis=1)
        Y = Y_unfold.reshape([code_num, -1])
        return Y

    def decode(self, X):
        code_num = len(X)
        r = self._n - self._k
        X_unfold = X.reshape([-1, self._n])
        S_unfold = (X_unfold @ self._H.T) % 2
        S_unfold = S_unfold @ 2 ** np.arange(r)[::-1]
        Y_unfold = np.logical_or(X_unfold[:, :4], self._C[S_unfold])
        Y = Y_unfold.reshape([code_num, -1])
        return Y


# haven't completed
class Cyclic:
    def __init__(self, n, k):
        self._n = n
        self._k = k
        self._gen_poly = cyclic_code_genpoly(n, k)[0]
        r = self._n - self._k

    def encode(self, X):
        code_num = len(X)
        X_encoded = [bit2dec(code) for code in X.reshape([-1, 4])] * self._gen_poly
        encoded_bits = np.array([dec2bit(bits, self._n) for bits in X_encoded])
        return encoded_bits.reshape([code_num, -1])

    def decode(self, X):
        code_num = len(X)


# modulation
# pulse forming filter
_, rcosfir = rcosfilter(N=8 * 8, alpha=0.5, Ts=1e-6, Fs=8e6)


def modulate(data, modulator, timing_err=False):
    signs_data = [modulator.modulate(bits) for bits in data]

    # oversampling (upsampling): actually filling "0"
    samp_signs_data = [upsample(signs, 8) for signs in signs_data]

    # pulse forming filter
    iq_data = [np.convolve(signs, rcosfir) for signs in samp_signs_data]

    # add timing error
    timing_offset = np.random.randint(0, 8, len(iq_data)) if timing_err else 0
    seq_start = len(rcosfir) // 2 * np.ones(len(iq_data), dtype=int) + timing_offset
    seq_end = seq_start + len(samp_signs_data[0])
    iq_data = np.array([iq[index[0]: index[1]] for iq, index in zip(iq_data, zip(seq_start, seq_end))])

    # orthogonal modulation
    n = np.repeat(np.arange(iq_data.shape[1]).reshape([1, -1]), iq_data.shape[0], 0)
    cos_carrier = np.cos(2 * np.pi / 8 * n)
    sin_carrier = np.sin(2 * np.pi / 8 * n)
    signal_data = iq_data.real * cos_carrier + iq_data.imag * sin_carrier
    return signal_data, iq_data, signs_data


# function convert Eb to SNR
def EbN02SNR(EbN0, sign_bit, samp_rate):
    EsN0 = EbN0 + 10*math.log10(sign_bit)
    SNR = EsN0 - 10*math.log10(samp_rate)
    return SNR


# generate the random sequence follow GGD(Generalized Gaussian Distribution)
def rand_ggd(rho, gamma, shape):
    # GGD的概率密度和概率分布
    x = np.arange(-10, 10, 0.02)
    ggd = (rho / (2 * gamma * math.gamma(1 / rho))) * np.exp(-(np.abs(x) / gamma) ** rho)
    ggd_sum = np.cumsum(ggd * 0.02)
    cdf_inv = interp.interp1d(ggd_sum, x, 'cubic')

    X = np.random.uniform(0, 1, *[shape])
    return cdf_inv(X)


# add noise to signal
def add_noise(signal, noise_type, snr):
    p_signal = (signal[0:100] ** 2).mean()
    n0 = p_signal / 10 ** (snr / 10)
    if noise_type == 'awgn':
        signal += np.random.randn(*signal.shape) * n0 ** 0.5
    elif noise_type == 'aggn':
        gain = n0 / (rand_ggd(1.5, 1, 1000) ** 2).mean()
        signal += rand_ggd(1.5, 1, signal.shape) * gain ** 0.5
    return signal


# traditional receiver
def old_receiver(input_data, modulator, decoder):

    # 载波生成
    n = np.repeat(np.arange(input_data.shape[1]).reshape([1, -1]), input_data.shape[0], 0)
    cos_carrier = np.cos(2*np.pi/8 * n)
    sin_carrier = np.sin(2*np.pi/8 * n)

    # 解出IQ信号
    lp_fir = firwin(100, 0.18) # 低通滤波器
    temp_data = (input_data*cos_carrier + input_data*sin_carrier*1j) * 2
    demod_data = np.array([np.convolve(temp, lp_fir, 'same') for temp in temp_data])

    # 匹配滤波器生成
    h = rcosfir[len(rcosfir)//2: len(rcosfir)//2 + 8]
    h_norm = h / np.linalg.norm(h)

    # 匹配滤波,符号判决,汉明解码
    recover_signs_data = (demod_data.reshape([len(demod_data), -1, 8]) * h_norm.reshape([1, 1, 8])).sum(2)
    receive_data = [modulator.demodulate(recover_signs, 'hard') for recover_signs in recover_signs_data]
    decoded_data = decoder.decode(np.array(receive_data))

    return decoded_data
