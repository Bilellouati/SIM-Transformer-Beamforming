import numpy as np
import math 
from scipy.io import savemat, loadmat

################################################## Function definition
# This function allows to map bitstream to complex symbols.
# It ONLY supports BPSK, QPSK, 16QAM, 64QAM, 256QAM, 1024QAM and PSK8 #
def mapping(bits, modulation_order, modulation_type):
    modulation_bits = int(math.log2(modulation_order))
    if modulation_bits == 1:
        if modulation_type == "ook":
            symb_mapping_cplx = bits*math.sqrt(2)
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
                gray_mapping = np.exp(np.array([0, 1, 3, 2, 6, 7, 5, 4])*1j*math.pi/4)
                for k in range(0, len(symb_mapping_tmp)):
                    for j in range(0, len(tab_mod)):
                        if symb_mapping_tmp[k] == tab_mod[j]:
                            symb_mapping_cplx[k] = gray_mapping[j]

        if modulation_type == "pam":
            symbs = bits.reshape(modulation_bits, int(n_bits / modulation_bits), order='F')
            pattern = np.ones((modulation_bits, int(n_bits / modulation_bits)))
            for k in range(modulation_bits):
                pattern[k,:] = int(math.pow(2, modulation_bits-1-k))
            
            symb_mapping = sum(symbs * pattern)
            
            if modulation_bits == 2:
                symb_mapping[:] = [0 if value == 0
                else 1 if value == 1
                else 2 if value == 3
                else 3 if value == 2
                else value for value in symb_mapping]
                symb_mapping_cplx = symb_mapping / math.sqrt(7/2)
                
        elif modulation_type == "qam":
            symbs = bits.reshape(int(modulation_bits / 2), int(n_bits / modulation_bits * 2), order='F')
            # QPSK
            if modulation_bits == 2:
                symb_mapping = np.squeeze(- (symbs * 2 - 1).T, axis=1)
                symb_mapping_cplx = (symb_mapping[0::2] + 1j * symb_mapping[1::2]) / math.sqrt(2)

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
                symb_mapping_cplx = (symb_mapping[0::2] + 1j * symb_mapping[1::2]) / math.sqrt(42)

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
                symb_mapping_cplx = (symb_mapping[0::2] + 1j * symb_mapping[1::2]) / math.sqrt(170)
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
                symb_mapping_cplx = (symb_mapping[0::2] + 1j * symb_mapping[1::2]) / math.sqrt(682)
    return symb_mapping_cplx


def frange(x, y, jump):
  while x < y:
    yield x
    x += jump
    

def randU(n):
    X = np.random.randn(n,n) + 1j*np.random.randn(n,n)
    Q, R = np.linalg.qr(X)
    R = np.diag(np.diag(R)/np.abs(np.diag(R)))
    return Q.dot(R)


def hpa_sspa_modif_rapp(vin, Vsat, p, q, G, A, B):
    a0 = np.abs(vin)
    theta = np.angle(vin)
    Am = (G * a0) / ((1 + (G * a0 / Vsat) ** (2 * p)) ** (1 / (2 * p)))
    Bm = (A * (a0 ** q)) / ((1 + (a0 / B) ** (q)))
    vout = Am * np.exp(1j * (theta + Bm))
    return vout


def find_K0_sigma2_d(IBO):
    xin = (1 / np.sqrt(2)) * (np.random.randn(1, 10000) + 1j * np.random.randn(1, 10000))
    coeff_IBO_m1dB = (
        val_IBO_sat * np.sqrt((1 / np.var(xin))) * np.sqrt(10 ** (-IBO / 10))
    )
    vin = coeff_IBO_m1dB * xin
    vout = hpa_sspa_modif_rapp(vin, Vsat, p, q, G, A, B)    
    K0 = np.mean(vout * np.conj(vin)) / np.mean(np.absolute(vin) ** 2)
    sigma2_d = np.mean(np.abs(vout - K0 * vin)**2)
    return (K0, sigma2_d)
       
################################################## System model
Mqam = 16
modulation_type = "qam"
Mfft = 256

### PA parameters : Modified Rapp model 
IBO = 3
IBOr= 10**(IBO/10)
p = 1.1
q = 4
Vsat = 1.9
G = 16
A = -345
B = 0.17
val_IBO_sat = Vsat/G
K0, sigma2_d = find_K0_sigma2_d(IBO)

### APs\UEs
L = 1     # Nbre of APs
M = 100    # Nbre of antennas per AP
K = 15    # Nbre of UEs
sigma_sh_dB = 4
sigma_sh = 10**(sigma_sh_dB/10)
Kau = 0.5
APh = 10
UEh = 1.5

### Pilot symbols 
Tau_p = K

### Block length
Tau_c = 200
xi = 0.5
#UOIindex = 19

### Transmit power is computed related to IBO and PA parameters
pin = (val_IBO_sat**2)/(10 ** (IBO / 10))
Pin = pin*M
Pinwm = 1000*Pin

### Noise power
wp_dBm = -92
wp = 0.001*(10**(wp_dBm/10))

### Total power budget at AP (DL Power control)
Rho_l_max_dBm = 10*np.log10(Pinwm)
Rho_l_max = 0.001*(10**(Rho_l_max_dBm/10))

### Power at UEs (UL channel estimation)
pk_dBm = 10*np.log10(100)
pk = 0.001*(10**(pk_dBm/10))

### Area dimension
D = 1

### Precoding scheme
prec = "FZF"

### Nbre of channel realizations
Nreal = 1000

### Initializations
WHk = np.zeros((L,Nreal),dtype=complex)
HD = np.zeros((L,Nreal),dtype=complex)
WHkt = np.zeros((L,Nreal,K),dtype=complex)
Qk = np.zeros((Nreal),dtype=complex)
Qkk = np.zeros((Nreal,K,Mfft),dtype=complex)
Xin = np.zeros((Nreal,M,L,Mfft),dtype=complex)
Xout = np.zeros((Nreal,M,L,Mfft),dtype=complex)
DisFreqR = np.zeros((Nreal,M,L,Mfft),dtype=complex)


################################################## Generate nodes
UOIindex = int(np.random.randint(0,K,1))
SCOIindex = int(np.random.randint(0,Mfft,1))
APs = np.zeros((L,3),dtype=float)
APs[:,0] = list(frange(1,L+1,1))
APs[:,1] = np.random.uniform(0,D,L)
APs[:,2] = np.random.uniform(0,D,L)

UEs = np.zeros((K,3),dtype=float)
UEs[:,0] = list(frange(1,K+1,1))
UEs[:,1] = np.random.uniform(0,D,K)
UEs[:,2] = np.random.uniform(0,D,K)

Dlk = np.zeros((L,K),dtype=float)
for l in range(0,L):
    for k in range(0,K):
        Dlk[l,k] = np.sqrt(np.abs(APs[l,1] - UEs[k,1])**2 + np.abs(APs[l,2] - UEs[k,2])**2 + np.abs(APh - UEh)**2)

### Path loss
PL_lk_dB = -30.5 - 36.7*np.log10(Dlk)
PL_lk = 10**(PL_lk_dB/10)
#sigma_sh_Zlk_dB = 4
al0 = np.random.randn(L,1)
al = np.repeat(al0,K,-1)
bk0 = np.random.randn(K,1)
bk00 = np.repeat(bk0,L,-1)
bk = np.transpose(bk00)
zlk = np.sqrt(Kau)*al + np.sqrt(1-Kau)*bk
sigma_sh_Zlk = sigma_sh*zlk
BETAlk = PL_lk*(10**(sigma_sh_Zlk/10))

################################################## Channel estimation
### Pilots generation
Pilots = np.sqrt(Tau_p)*randU(Tau_p)

### Uniform Pilot assignement
userPilotIndex0 = list(frange(0,Tau_p,1))
userPilotIndex00 = [] 
for tp in range(0,int(K/Tau_p)+1):
    userPilotIndex00 = np.concatenate((userPilotIndex00,userPilotIndex0))
userPilotIndex = userPilotIndex00[:K]
PhiP = np.zeros((Tau_p,K),dtype=complex)
for up in range(0,K):
    PhiP[:,up] = Pilots[:,int(userPilotIndex[up])]
    
### Power at the UEs
Pk = pk*np.diag(np.ones(K))

### Channel estimation
Clk = np.zeros((L,K),dtype=float)
for k in range(0,K):
    UONIindex = [] 
    for kk in range(0,K):
        if (userPilotIndex[kk]==userPilotIndex[k]) & (kk!=k):
            UONIindex = np.concatenate((UONIindex,np.reshape(kk,(1))),0)
    for l in range(0,L):
        betalt = 0
        for ind in range(0,len(UONIindex)):
            betalt = betalt + BETAlk[l,int(UONIindex[ind])]
        Clk[l,k] = (np.sqrt(pk)*BETAlk[l,k])/((Tau_p*pk*(betalt+BETAlk[l,k])) + wp)

NUlk = np.zeros((L,K),dtype=float)
for k in range(0,K):
    UONIindex = [] 
    for kk in range(0,K):
        if (userPilotIndex[kk]==userPilotIndex[k]) & (kk!=k):
            UONIindex = np.concatenate((UONIindex,np.reshape(kk,(1))),0)
    for l in range(0,L):
        betalt = 0
        for ind in range(0,len(UONIindex)):
            betalt = betalt + BETAlk[l,int(UONIindex[ind])]
        NUlk[l,k] = ((pk*Tau_p)*(BETAlk[l,k]**2))/((pk*Tau_p*(betalt+BETAlk[l,k])) + wp)
        
Thetalk = NUlk/(Clk**2)
        
UONIindex = [] 
for kk in range(0,K):
    if (userPilotIndex[kk]==userPilotIndex[UOIindex]) & (kk!=UOIindex):
        UONIindex = np.concatenate((UONIindex,np.reshape(kk,(1))),0)
        
betalt = 0
for ind in range(0,len(UONIindex)):
    betalt = betalt + BETAlk[l,int(UONIindex[ind])]

### DL Power allocation model 
RHOlk = np.zeros((L,K),dtype=float)
for l in range(0,L):
    for k in range(0,K):
        RHOlk[l,k] = ((NUlk[l,k]/np.sum(NUlk[l,:]))*Rho_l_max)     

for iNreal in range(0,Nreal):    
    H3 = (1 / np.sqrt(2)) * (np.random.randn(M, K, L) + 1j * np.random.randn(M, K, L))
    ### Uplink Training phase
    realH3 = np.zeros((M,K,L),dtype=complex)
    for l in range(0,L):
        realH3[:,:,l] = H3[:,:,l].dot(np.diag(np.sqrt(BETAlk[l,:])))
    
    ### Channel in frequency domain
    realHFreq3 = np.zeros((M,K,L,Mfft),dtype=complex)
    for l in range(0,L):
        for k in range(0,K):
            for m in range(0,M):
                realHFreq3[m,k,l,:] = np.fft.fft(np.concatenate((np.reshape(realH3[m,k,l],(1,1)),np.zeros((1,Mfft-1),dtype=complex)),1))
    
    ### Perfect channel True/False
    Yl = np.zeros((M,Tau_p,L,Mfft),dtype=complex)
    for l in range(0,L):
        for mf in range(0,Mfft):
            Nl = np.sqrt(wp/(2))*(np.random.randn(M,Tau_p) + 1j*np.random.randn(M,Tau_p))
            Yl[:,:,l,mf] = realHFreq3[:,:,l,mf].dot(np.sqrt(Pk).dot(np.conjugate(np.transpose(PhiP)))) + Nl 

    
    H3barl = np.zeros((M,Tau_p,L,Mfft),dtype=complex)
    for l in range(0,L):
        for mf in range(0,Mfft):
            H3barl[:,:,l,mf] = Yl[:,:,l,mf].dot(Pilots)

################################################## Downlink Data Transmission
    ### FZF Precoding
    W_FZFl = np.zeros((M,Tau_p,L,Mfft),dtype=complex)
    for l in range(0,L):
        for mf in range(0,Mfft):
            W_FZFl[:,:,l,mf] =  H3barl[:,:,l,mf].dot(np.linalg.inv(np.transpose(np.conjugate(H3barl[:,:,l,mf])).dot(H3barl[:,:,l,mf])))

    for l in range(0,L):
        for k in range(0,Tau_p):
            for mf in range(0,Mfft):
                W_FZFl[:,k,l,mf] = np.sqrt((M-Tau_p)*Thetalk[l,k])*W_FZFl[:,k,l,mf]

    
    WW = np.zeros((M,(int(K/Tau_p)+1)*Tau_p,L,Mfft),dtype=complex)
    for tp in range(0,int(K/Tau_p)+1):
        WW[:,tp*Tau_p:(tp+1)*Tau_p,:,:] = W_FZFl

    W_FZFlk = WW[:,:K,:]
    Wlk = W_FZFlk
    
    ### Bit generation
    bitk = np.random.randint(2,size=int(np.log2(Mqam)*K*Mfft))
    qk = mapping(bitk, Mqam, modulation_type)
    qk = qk/np.sqrt(np.var(qk))
    qkP = np.reshape(qk,(K,Mfft)) 
    Qk[iNreal] = qkP[UOIindex,SCOIindex] 
    Qkk[iNreal,:,:] = qkP
    
    Xl = np.zeros((M,L,Mfft),dtype=complex)
    xlOFDM = np.zeros((M,L,Mfft),dtype=complex)
    XRl = np.zeros((M,L,Mfft),dtype=complex)
    DisFreq = np.zeros((M,L,Mfft),dtype=complex)
    for l in range(0,L):
        for mf in range(0,Mfft):
            ### Precoding
            Xl[:,l,mf] = Wlk[:,:,l,mf].dot(np.diag(np.sqrt(RHOlk[l,:]))).dot(qkP[:,mf])
            Xin[iNreal,:,l,mf] = Xl[:,l,mf]
        ### OFDM modulation 
        xlOFDM[:,l,:] = np.sqrt(Mfft)*np.fft.ifft(Xl[:,l,:])
        
    xlOFDM_amp = np.zeros((M,L,Mfft),dtype=complex)
    disT = np.zeros((M,L,Mfft),dtype=complex)
    for l in range(0,L):    
        for m in range(0,M):
            ### PA amplification
            vin =  xlOFDM[m,l,:]
            xlOFDM_amp[m,l,:] = hpa_sspa_modif_rapp(vin, Vsat, p, q, G, A, B)
            ### Distortion term
            disT[m,l,:] = xlOFDM_amp[m,l,:] - K0*vin
        # OFDM demodulation
        XRl[:,l,:] =  np.sqrt(1/Mfft)*np.fft.fft(xlOFDM_amp[:,l,:])
        DisFreq[:,l,:] = np.sqrt(1/Mfft)*np.fft.fft(disT[:,l,:])
        
    ### Output of the PA
    for l in range(0,L):    
        for m in range(0,M):
            Xout[iNreal,m,l,:] = XRl[m,l,:]
            DisFreqR[iNreal,:,l,:] = DisFreq[:,l,:]
            
    for l in range(0,L):
        WHk[l,iNreal] = np.conjugate(np.transpose(realHFreq3[:,UOIindex,l,SCOIindex])).dot(Wlk[:,UOIindex,l,SCOIindex])*np.sqrt(RHOlk[l,UOIindex])
        HD[l,iNreal] = np.reshape(np.conjugate(np.transpose(realHFreq3[:,UOIindex,l,SCOIindex])),(1,M)).dot(DisFreq[:,l,SCOIindex])

    for l in range(0,L):
        for tU in range(0,K):
            if (tU != UOIindex):
                WHkt[l,iNreal,tU] =  np.conjugate(np.transpose(realHFreq3[:,UOIindex,l,SCOIindex])).dot(Wlk[:,tU,l,SCOIindex])*np.sqrt(RHOlk[l,tU])

################################################## SINR computation
CPk0 = 0
PUk0 = 0
UIkt0 = 0
HD0 = 0
for l in range(0,L):
    CPk0 = CPk0 + np.mean(K0*WHk[l,:])
    PUk0 = PUk0 + K0*WHk[l,:] - np.mean(K0*WHk[l,:])
    UIkt0 = UIkt0 + K0*WHkt[l,:,:]
    HD0 = HD0 + (HD[l,:])

t1 = np.zeros((Nreal),dtype=complex)
t2 = np.zeros((Nreal),dtype=complex)
t3 = np.zeros((Nreal),dtype=complex)
t4 = np.zeros((Nreal),dtype=complex)
t5 = np.zeros((Nreal),dtype=complex)
for iNreal in range(Nreal):
    t1[iNreal] = CPk0*Qk[iNreal]
    t2[iNreal] = PUk0[iNreal]*Qk[iNreal]
    t3[iNreal] = np.sum([UIkt0[iNreal,ik]*Qkk[iNreal,ik,SCOIindex] for ik in range(K) if ik!=UOIindex]) 
    nk = np.sqrt(wp/(2))*(np.random.randn(K,Mfft) + 1j*np.random.randn(K,Mfft))
    t4[iNreal] = nk[UOIindex,SCOIindex]
    t5[iNreal] = HD0[iNreal]

################################################## Correlationbetween Xin, Xout and Distortion
Rout = np.zeros((Nreal,M,M),dtype=complex)
Rd = np.zeros((Nreal,M,M),dtype=complex)
Rin = np.zeros((Nreal,M,M),dtype=complex)
for iNreal in range(Nreal):
    A = Xin[iNreal,:,0,SCOIindex].reshape(M,1)
    B = Xout[iNreal,:,0,SCOIindex].reshape(M,1)
    C = DisFreqR[iNreal,:,0,SCOIindex].reshape(M,1)
    Rout[iNreal,:,:] = A.dot(np.conjugate(B).T)
    Rd[iNreal,:,:] = A.dot(np.conjugate(C).T)
    Rin[iNreal,:,:] = A.dot(np.conjugate(A).T)
    
Corr_Xin_Xout = np.mean(Rout,axis=0)
Corr_Xin_D = np.mean(Rd,axis=0)
Corr_Xin_Xin = np.mean(Rin,axis=0)

################################################## Correlation between terms in SINR 
Termes = np.array([t1, t2, t3, t4, t5])
meanR = np.zeros((Nreal,5,5),dtype=complex)
for iNreal in range(Nreal):
    for i in range(5):
        A = Termes[i,iNreal]
        for j in range(5):
            B = Termes[j,iNreal]
            meanR[iNreal,i,j] =  A*np.conjugate(B)

Corr_Termes = np.mean(meanR,axis=0)
