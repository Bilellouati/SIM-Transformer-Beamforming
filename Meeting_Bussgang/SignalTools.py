import numpy as np
import math as m

# The function below generate an awgn channel using a signal x #
import scipy.special


def awgn_channel_generator(x, snr, modulation_type, pn_in=0, ps_in=0):
    if modulation_type == "ook" or modulation_type == "qpsk" or modulation_type == "pam":
        noise = np.random.randn(len(x)) * (10 ** (-snr / 20))
    else:
        noise = (np.random.randn(len(x)) + 1j * np.random.randn(len(x))) * (m.sqrt(2)/2) * 10 ** (-snr / 20)
    ps = ps_in + np.mean(np.abs(x) ** 2)
    pn = pn_in + np.mean(np.abs(noise) ** 2)
    y = x + noise
    return [y, pn, ps]


def hard_decoding(s_rec, modulation_type, modulation_order):
    s_rec_tmp = np.sqrt(s_rec)*m.sqrt(7/2)
    s_rec_den = np.sqrt(s_rec)*m.sqrt(7/2)
    # PAM-4 thresholds
    # 0 ----- 1 ----- 2 ----- 3
    if modulation_type == "pam":
        s_rec_den[(s_rec_tmp>=2.5)]=2
        s_rec_den[(s_rec_tmp >= 1.5) & (s_rec_tmp < 2.5)]=3
        s_rec_den[(s_rec_tmp >=0.5) & (s_rec_tmp<1.5)]=1
        s_rec_den[s_rec_tmp <0.5] = 0
    
    bits = np.array(list(map('{:02b}'.format, s_rec_den.astype(int))))
    bits = np.array(list(map(list, bits))).reshape(1,-1)
    bits = np.array(list(map(int, bits[0]))).astype(float)
    return bits

def hard_demapping(s_rec, modulation_type, modulation_order):
    mapped, bits_mapped = get_symbol_mapping(modulation_order, modulation_type)
    #idxs = np.zeros((len(s_rec), 1))
    bits = np.zeros((len(s_rec), len(bits_mapped[0])))

    for k in range(len(s_rec)):
        mapped_tmp = mapped
        mapped_tmp = np.abs(mapped_tmp - s_rec[k])
        idx = np.argmin(mapped_tmp)
        bits[k ,:] = bits_mapped[idx, :]

    bits = bits.reshape(1, int(len(s_rec)*len(bits_mapped[0])))
    bits = bits.squeeze().astype(int)
    return bits

# def apply_detection_law(s_rec):
#     file_path = str(Path.home()) + "/PythonProjects/llrnet_ONLY/in_out_law.mat"
#     p = loadmat(file_path)
#     p_values = p['p'].tolist()[0]
#     return np.abs(np.polyval(p_values, s_rec)+0.015)


# This function allows to map bitstream to complex symbols.
# It ONLY supports BPSK, QPSK, 16QAM, 64QAM, 256QAM, 1024QAM and PSK8 #
def mapping(bits, modulation_order, modulation_type):
    modulation_bits = int(m.log2(modulation_order))
    if modulation_bits == 1:
        if modulation_type == "ook":
            symb_mapping_cplx = bits*m.sqrt(2)
        else:
            symb_mapping_cplx = -(bits * 2 - 1)

    else:
        n_bits = len(bits)

        if modulation_type == "psk":
            symbs = bits.reshape(modulation_bits, int(n_bits / modulation_bits), order='F')
            if modulation_bits == 3:
                tab_mod = [0, 1, 3, 2, 6, 7, 5, 4]
                symb_mapping_tmp = symbs[0, :]*4 + symbs[1, :]*2 + symbs[2, :]*1
                symb_mapping_cplx = np.zeros(int(n_bits / modulation_bits)).astype(complex)
                gray_mapping = np.exp(np.array([0, 1, 3, 2, 6, 7, 5, 4])*1j*m.pi/4)
                for k in range(0, len(symb_mapping_tmp)):
                    for j in range(0, len(tab_mod)):
                        if symb_mapping_tmp[k] == tab_mod[j]:
                            symb_mapping_cplx[k] = gray_mapping[j]

        if modulation_type == "pam":
            symbs = bits.reshape(modulation_bits, int(n_bits / modulation_bits), order='F')
            pattern = np.ones((modulation_bits, int(n_bits / modulation_bits)))
            for k in range(modulation_bits):
                pattern[k,:] = int(m.pow(2, modulation_bits-1-k))
            
            symb_mapping = sum(symbs * pattern)
            
            if modulation_bits == 2:
                symb_mapping[:] = [0 if value == 0
                else 1 if value == 1
                else 2 if value == 3
                else 3 if value == 2
                else value for value in symb_mapping]
                symb_mapping_cplx = symb_mapping / m.sqrt(7/2)
                
        elif modulation_type == "qam":
            symbs = bits.reshape(int(modulation_bits / 2), int(n_bits / modulation_bits * 2), order='F')
            # QPSK
            if modulation_bits == 2:
                symb_mapping = np.squeeze(- (symbs * 2 - 1).T, axis=1)
                symb_mapping_cplx = (symb_mapping[0::2] + 1j * symb_mapping[1::2]) / m.sqrt(2)

            # QAM16
            elif modulation_bits == 4:
                # ------------------------- #
                #    |    |     |     |
                #   10   11    01    00
                #   -3   -1    +1    +3
                pattern = np.ones((2, int(n_bits / modulation_bits * 2)))
                pattern[0, :] = 2
                symb_mapping = sum(symbs * pattern)
                symb_mapping[:] = [3 if value == 0
                                   else 1 if value == 1
                else -3 if value == 2
                else -1 if value == 3
                else value for value in symb_mapping]
                symb_mapping_cplx = (symb_mapping[0::2] + 1j * symb_mapping[1::2])

            # QAM 64
            elif modulation_bits == 6:
                # -------------------------------------------------- #
                #  |     |      |      |     |      |      |      |
                # 100   101    111    110   010    011    001   000
                # -7    -5     -3     -1    +1     +3     +5    +7
                pattern = np.ones((3, int(n_bits / modulation_bits * 2)))
                pattern[0, :] = 4
                pattern[1, :] = 2
                pattern[2, :] = 1
                symb_mapping = sum(symbs * pattern)
                symb_mapping[:] = [7 if value == 0
                                   else 5 if value == 1
                else 1 if value == 2
                else 3 if value == 3
                else -7 if value == 4
                else -5 if value == 5
                else -1 if value == 6
                else -3 if value == 7
                else value for value in symb_mapping]
                symb_mapping_cplx = (symb_mapping[0::2] + 1j * symb_mapping[1::2]) / m.sqrt(42)

            # QAM 256
            elif modulation_bits == 8:
                tab_mod = [15, 13, 9, 11, 1, 3, 7, 5, - 15, - 13, - 9, - 11, - 1, - 3, - 7, - 5]
                pattern = np.ones((4, int(n_bits / modulation_bits * 2)))
                pattern[0, :] = 8
                pattern[1, :] = 4
                pattern[2, :] = 2
                pattern[3, :] = 1
                symb_mapping_tmp = sum(symbs * pattern)
                symb_mapping = np.zeros((int(n_bits / modulation_bits * 2)))
                for i in range(0, 16):
                    for j in range(0, len(symb_mapping_tmp)):
                        if symb_mapping_tmp[j] == i:
                            symb_mapping[j] = tab_mod[i]
                symb_mapping_cplx = (symb_mapping[0::2] + 1j * symb_mapping[1::2]) / m.sqrt(170)
            # QAM 1024
            elif modulation_bits == 10:
                tab_mod = [31, 29, 25, 27, 17, 19, 23, 21, 1, 3, 7, 5, 15, 13, 9, 11,
                           -31, -29, -25, -27, -17, -19, -23, -21, -1, -3, -7, -5, -15, -13, -9, -11]
                pattern = np.ones((5, int(n_bits / modulation_bits * 2)))
                pattern[0, :] = 16
                pattern[1, :] = 8
                pattern[2, :] = 4
                pattern[3, :] = 2
                pattern[4, :] = 1
                symb_mapping_tmp = sum(symbs * pattern)
                symb_mapping = np.zeros((int(n_bits / modulation_bits * 2)))
                for i in range(0, 32):
                    for j in range(0, len(symb_mapping_tmp)):
                        if symb_mapping_tmp[j] == i:
                            symb_mapping[j] = tab_mod[i]
                symb_mapping_cplx = (symb_mapping[0::2] + 1j * symb_mapping[1::2]) / m.sqrt(682)
    return symb_mapping_cplx


# This function will generate a bitstream using a random seed #
def binary_stream_generator(bits_number, seed):
    if seed == 1:
        bits = np.zeros(1, bits_number)
        print("binary_source: all zeros vector is generated")
    elif seed == -1:
        rng = np.random.default_rng()
        bits = np.floor(2*np.random.rand(1, bits_number))
        rng.shuffle(bits)
    else:
        s = np.random.RandomState(seed)
        bits = s.randint(2, size=bits_number)
    return bits

def pdf_chi2(s_rec, c_symbs, sigma2):
    M = 1/2
    pdf = 0
    for value in c_symbs:
        if value == 0:
            pdf = np.exp(-s_rec/(2*sigma2))* (s_rec)**(M-1) * 1/(m.sqrt(2*sigma2)*scipy.special.gamma(M))
        else:
            pdf = scipy.special.iv(M-1, 2*np.sqrt(s_rec*value**2)/(2*sigma2)) * np.exp(-(s_rec+value**2)/(2*sigma2)) * (s_rec/value**2)**((M-1)/2) * 1/(2*sigma2)
        pdf += pdf
    return pdf

# This function allows to compute LLR using max-log or exact formula for given modulation order and SNRdB #
def compute_llrs(method, modulation_order, modulation_type, s_rec, snr):
    sigma2 = 10**(-snr/10)
    M = 1/2
    modulation_nbits = int(m.log2(modulation_order))
    symbs, bit_array = get_symbol_mapping(modulation_order, modulation_type)

    symbs_re = np.real(symbs)
    symbs_imag = np.imag(symbs)

    s_rec_re = np.real(s_rec)
    s_rec_imag = np.imag(s_rec)

    # Define constellation bits #
    c0 = np.zeros((int(2**modulation_nbits / 2), modulation_nbits)).astype(complex)
    c1 = np.zeros((int(2 ** modulation_nbits / 2), modulation_nbits)).astype(complex)
      

    for k in range(0, modulation_nbits):
        if modulation_type == "qam":
            if k < int(modulation_nbits / 2):
                c0[:, k] = symbs_re[np.nonzero(bit_array[:, k] - 1)]
                c1[:, k] = symbs_re[np.nonzero(bit_array[:, k])]
            else:
                c0[:, k] = symbs_imag[np.nonzero(bit_array[:, k] - 1)]
                c1[:, k] = symbs_imag[np.nonzero(bit_array[:, k])]
        else:
            c0[:, k] = symbs[np.nonzero(bit_array[:, k] - 1)]
            c1[:, k] = symbs[np.nonzero(bit_array[:, k])]

    llrs = np.zeros((modulation_nbits, len(s_rec)))
    
    if modulation_type == "ook":
        num = np.exp(-s_rec/(2*sigma2)) * (s_rec)**(M-1) * 1/(m.sqrt(2*sigma2)*scipy.special.gamma(M))
        den = scipy.special.iv(M-1,2*np.sqrt(s_rec*2)/(2*sigma2)) * np.exp(-(s_rec+2)/(2*sigma2)) * (s_rec/2)**((M-1)/2) * 1/(2*sigma2)
        llrs[0, :] = np.log(num) - np.log(den)

    if modulation_type == "pam":
        c0 = np.real(c0)
        c1 = np.real(c1)
            
        for k in range(0, modulation_nbits):
            num = pdf_chi2(s_rec, c0[:, k], sigma2)
            den = pdf_chi2(s_rec, c1[:, k], sigma2)
            llrs[k, :] = np.log(num) - np.log(den)      

    elif method == "maxlog":
        for k in range(0, modulation_nbits):
            if modulation_type == "qam":
                if k < int(modulation_nbits / 2):
                    value = s_rec_re
                else:
                    value = s_rec_imag
            else:
                value = s_rec
            distc1 = np.abs(np.tile(value, (int(2 ** modulation_nbits / 2), 1))
                            - np.tile(c1[:, k], (len(value), 1)).T) ** 2
            distc0 = np.abs(np.tile(value, (int(2 ** modulation_nbits / 2), 1))
                            - np.tile(c0[:, k], (len(value), 1)).T) ** 2
            llrs[k, :] = (1/sigma2)*(distc1.min(0) - distc0.min(0))
    elif method == "logmap":
        for k in range(0, modulation_nbits):
            num = np.exp(- np.abs(np.tile(s_rec, (int(2 ** modulation_nbits / 2), 1))
                                  - np.tile(c0[:, k], (len(s_rec), 1)).T) ** 2 / sigma2)
            den = np.exp(- np.abs(np.tile(s_rec, (int(2 ** modulation_nbits / 2), 1))
                                  - np.tile(c1[:, k], (len(s_rec), 1)).T) ** 2 / sigma2)

            llrs[k, :] = np.log(sum(num)) - np.log(sum(den))

    return llrs


def compute_likelihoods(method, modulation_order, modulation_type, s_rec, snr):
    sigma2 = 10**(-snr/10)
    M = 1/2
    modulation_nbits = int(m.log2(modulation_order))
    symbs, bit_array = get_symbol_mapping(modulation_order, modulation_type)
    symbs_re = np.real(symbs)
    symbs_imag = np.imag(symbs)

    s_rec_re = np.real(s_rec)
    s_rec_imag = np.imag(s_rec)

    #symbs = np.imag(symbs)
    #s_rec = np.imag(s_rec)

    # Define constellation bits #
    c0 = np.zeros((int(2**modulation_nbits / 2), modulation_nbits)).astype(complex)
    c1 = np.zeros((int(2 ** modulation_nbits / 2), modulation_nbits)).astype(complex)

    for k in range(0, modulation_nbits):
        if modulation_type == "qam":
            if k < int(modulation_nbits/2):
                c0[:, k] = symbs_re[np.nonzero(bit_array[:, k] - 1)]
                c1[:, k] = symbs_re[np.nonzero(bit_array[:, k])]
            else:
                c0[:, k] = symbs_imag[np.nonzero(bit_array[:, k] - 1)]
                c1[:, k] = symbs_imag[np.nonzero(bit_array[:, k])]
        else:
            c0[:, k] = symbs[np.nonzero(bit_array[:, k] - 1)]
            c1[:, k] = symbs[np.nonzero(bit_array[:, k])]

    lls = np.zeros((2, modulation_nbits, len(s_rec)))

    if modulation_type == "ook":
        num = np.exp(-s_rec/(2*sigma2)) * (s_rec/(2*sigma2))**(M-1) * 1/((2*sigma2)*scipy.special.gamma(M))
        den = scipy.special.iv(M-1,2*np.sqrt(s_rec*2)/(2*sigma2)) * np.exp(-(s_rec+2)/(2*sigma2)) * (s_rec/2)**((M-1)/2) * 1/(2*sigma2)
        lls[0, 0, :] = num
        lls[1, 0, :] = den

    elif method == "maxlog":
        for k in range(0, modulation_nbits):
            distc1 = np.abs(np.tile(s_rec, (int(2 ** modulation_nbits / 2), 1))
                            - np.tile(c1[:, k], (len(s_rec), 1)).T) ** 2
            distc0 = np.abs(np.tile(s_rec, (int(2 ** modulation_nbits / 2), 1))
                            - np.tile(c0[:, k], (len(s_rec), 1)).T) ** 2
            lls[k, :] = (1/sigma2)*(distc1.min(0) - distc0.min(0))
    elif method == "logmap":
        for k in range(0, modulation_nbits):
            if modulation_type == "qam":
                if k < int(modulation_nbits / 2):
                    value = s_rec_re
                else:
                    value = s_rec_imag
            else:
                value = s_rec
            num = np.exp(- np.square(np.abs(np.tile(value, (int(2 ** modulation_nbits / 2), 1))
                                            - np.tile(c0[:, k], (len(s_rec), 1)).T)) / sigma2)
            den = np.exp(- np.square(np.abs(np.tile(value, (int(2 ** modulation_nbits / 2), 1))
                                            - np.tile(c1[:, k], (len(s_rec), 1)).T)) / sigma2)

            lls[0, k, :] = sum(num)
            lls[1, k, :] = sum(den)
    return lls


def compute_llrs_from_lls(lls):
    #ipdb.set_trace()
    epsilon = 1e-10
    lls = np.log(lls+epsilon)
    llrs = lls[0, :, :] - lls[1, :, :]
    return llrs


def reformat_llrs_bch(llrs, modulation_order):
    llrs = llrs.reshape((-1, 1), order='F')
    #llrs = llrs.reshape((int(llrs.shape[0] / int(m.log2(modulation_order))),  int(m.log2(modulation_order))))
    llrs = llrs.reshape((int(llrs.shape[0] / 63), 63))
    llrs[llrs == np.inf] = 70
    llrs[llrs == -np.inf] = -70
    return llrs

def reformat_llrs(llrs):
    llrs = llrs.reshape((-1, 1), order='F')
    llrs[llrs == np.inf] = 1
    llrs[llrs == -np.inf] = -1
    return llrs

# Allows to get the symbols mapping for a given modulation order #
def get_symbol_mapping(modulation_order, modulation_type):
    modulation_bits = int(m.log2(modulation_order))
    values = np.arange(0, 2**modulation_bits)
    # Format is used to display a value according to a specific format (binary, hexadecimal, and much more) #
    bits = [format(value, '0' + str(modulation_bits) + 'b') for value in values]
    bit_array = np.array([list(value) for value in bits]).astype(int)
    bit_list = [int(bit) for bit in list(''.join(bits))]
    return mapping(np.array(bit_list), modulation_order, modulation_type), bit_array


# Retrieve bit sequence from llrs #
def get_bits_from_llrs(llrs, bits_number):
    bits = np.abs(np.sign(llrs) - 1) / 2
    bit_est = bits.reshape(1, bits_number, order='F')
    return bit_est.astype(int)
