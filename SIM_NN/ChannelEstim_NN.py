###### Importing modules
from functions import *

###### Defining parameters
### Users
K = 4 #users
d_UE = 10
PT = 10 #dBm
PTl = 0.001*(10**(PT/10))
UE_antGain = 0 #dBi
### BS
M = K #antennas
H_BS = 10
BSantGain = 5 # dBi
BSantGainL = 10**(BSantGain/10)
budget = M*BSantGainL*PTl
### SIM
L = 2 #meatsurfaces
Nx = 7
Ny = 7
N = Nx*Ny #meta-atoms
NbitQ = 2
c = 3e8
f = 28e9 #Hz
Bb = 10e6 #Hz
lamda = c/f
dx = lamda/2
dy = lamda/2
Tsim = 5*lamda
d_layer = Tsim/L
Delta_theta = (2*np.pi)/(2**NbitQ)
### Noise
sigma2k = -104 #dBm
wp = 0.001*(10**(sigma2k/10)) # mw
### Path-Loss
d0 = 1 # Reference distance
nbar = 3.5 # pathloss exponnet 
C0 = 10**(-40/10) # -40dB
#C0 = (lamda/(4*np.pi*d0))**2 #free space pathloss
# Noise power
pwdb = -120 #92 #dbm (sigma)
sigma = pow(10,-3)*pow(10,pwdb/10) #mwatt

###### Neural Network parameter
### Batch size
btch = 32 #32
### Channel realizations
Nch = 1000 #30*btch

# Custom activation function to map values to the range [0, 2*pi]
def custom_activation_phase(x):
    return tf.math.scalar_mul(2*np.pi, tf.keras.activations.sigmoid(x))


def custom_activation_power(x, threshold=budget):
    # Ensure the outputs are positive
    x = tf.nn.relu(x)
    # Apply a constraint to ensure the sum of the outputs is less than or equal to the threshold
    constrained_output = x / tf.reduce_sum(x, axis=-1, keepdims=True) * threshold
    return constrained_output

###### Computing distances
### Between meta_atoms of a layer
Dnn = distance_dnn(N,Nx,lamda)
### Between the (l-1)-th meta-surface and the l-th meta-surface
Dl = distance_layers(L,N,d_layer,Dnn)

###### Angle between propagdirection and normal direction
cosXl = np.zeros((L,N,N),dtype=float)
for l in range(L):
    for n in range(0,N):
        for npr in range(0,N):
            cosXl[l,n,npr] = d_layer/Dl[l,n,npr]

###### Transmission matrix from layer (l-1) to l-th layer
Wl = transmission_matrix(L,N,Dl,dx,dy,cosXl,lamda)

###### Transmission vector from BS to the first layer
### Distance from BS to the first layer
D1 = distance_BS_Layer(N,M,Nx,Ny,lamda,d_layer)
### Transmission vector
W1 = transmission_vector(N,M,d_layer,dx,dy,D1,lamda)

###### Link distances from the SIM to the UEs & pathlosses
### Path-loss
Betak = pathloss(K,H_BS,Tsim,d_UE,C0,d0,nbar)
### Correlation matrix
Rnn = correlation_matrix(N,Dnn,lamda)

###### Neural Network
### Prepare dataset
### Genarate channel matrix used for training & validation
H,_ = generate_channels(K,N,Nch,Betak,Rnn)
Htrain0 = np.reshape(H,(K*N,Nch))
Htrain0r = np.real(Htrain0)
Htrain0i = np.imag(Htrain0)
Htrain = np.concatenate((Htrain0r,Htrain0i),axis=0).transpose()

### Received pilot signal
# Intialize all shifts
thetas = np.zeros((L,N),dtype=np.float32)
thetas = np.random.uniform(low=0, high=2*np.pi,size=(L,N))
B = np.array([i*Delta_theta for i in range((2**NbitQ))])

for l in range(L):
    for n in range(N):
        indx = np.argmin(abs(thetas[l,n]-B))
        thetas[l,n] = B[indx]
PHI = np.exp(0+1j*thetas) 

# Forward propagation G
G = np.diag(PHI[0,:])
for l in range(1,L):
    G = np.matmul(np.matmul(np.diag(PHI[l,:]),Wl[l,:,:]),G)

# Noise Generation
Nl=sqrt(0.5*sigma)*np.array([[[(np.random.random()+np.random.random()*1j) for _ in range(Nch)] for _ in range(K)] for _ in range(N)])

# Pilots' generation
tau_p = 7
pilots = np.array([[[(np.random.random()+np.random.random()*1j) for _ in range(N)] for _ in range(K)] for _ in range(Nch)])

yp = np.zeros((K,N,Nch),dtype=complex)
for i in range(Nch):
    for k in range(K):
        yp[k,:,i] = (np.conjugate(H[k].T).dot(G.dot(sum([W1[:,k].reshape(N,1).dot(pilots[i,k].reshape(1,N)) for k in range(K)])+Nl[:,k,i].reshape(N,1)))[0,0]
        )
yp0 = np.reshape(yp,(K*N,Nch))
yp0r = np.real(yp0)
yp0i = np.imag(yp0)
yptrain = np.concatenate((yp0r,yp0i),axis=0).transpose()

#### NN Model
Input = Input(shape=(2*K*N,))
input_layer = Dense(2*K*N, activation='relu')(Input)
layer1 = Dense(2*K*N, activation='relu')(input_layer)
layer2 = Dense(2*K*N, activation='relu')(layer1)
Output = Dense(2*K*N, activation='relu')(layer2)
# Create the model
SIMnetwork = Model(inputs=Input, outputs=Output)
# Compile the model with appropriate loss functions
SIMnetwork.compile(optimizer='adam', loss='mse', metrics=['mse'])
print (SIMnetwork.summary())
### Fit the model : returns a history object
history = SIMnetwork.fit(x=yptrain, y=Htrain, validation_data=(yptrain, Htrain), epochs=500, batch_size=btch)

### Plot
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.grid(True,which="both", linestyle='--')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Val'], loc='upper right')
plt.show()