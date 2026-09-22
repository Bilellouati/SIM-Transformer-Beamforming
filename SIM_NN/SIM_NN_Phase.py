###### Importing modules
from functions import *
from tensorflow.keras.optimizers import SGD
from keras.layers import Dropout

def NormP_Q(sig):
    # Phase shift matrix 
    ypredQ =  tf.quantization.fake_quant_with_min_max_args(sig,0,((2**NbitQ)-1)*Delta_theta,NbitQ)
    return ypredQ

def CustomLoss(y_true,y_pred): 
    #Ht = tf.convert_to_tensor(H, dtype=tf.complex64)
    # print("Shape y_true: ",y_true.shape)
    # print("y_true: ",y_true)
    # print("y_true: ",y_pred)
    # print("Shape H: ",H1.shape)
    # Retrieve Htrain0r and Htrain0i
    num_rows = (2*K*N) // 2
    Htrain0r_retrieved = y_true[:,:num_rows].transpose()
    Htrain0i_retrieved = y_true[:,num_rows:].transpose()
    Hri = Htrain0r_retrieved + 1j* Htrain0i_retrieved
    print("Shape Hri: ",Hri.shape)
    Ht = tf.reshape(Hri,(K,N,btch))
    # Phase shift matrix Quantization
    ypredQ =  tf.quantization.fake_quant_with_min_max_args(y_pred,0,((2**NbitQ)-1)*Delta_theta,NbitQ)
    y_predt = tf.math.exp(tf.complex(0.0,ypredQ))
    #y_predt = tf.math.exp(Delta_theta*tf.complex(0.0, y_pred//Delta_theta))
    PHIt = tf.reshape(y_predt,(btch,L,N))
    rho = [Betak[k,0]/sum(Betak[:,0]) for k in range(K)]
    Puet = np.array([[budget/K for k in range(K)] for _ in range(btch)],dtype=np.float64)
    #Puet = np.array([[budget*Betak[k]/sum(Betak) for k in range(K)] for _ in range(btch)])
    sumR = tf.zeros((btch,1),dtype=np.float64)
    
    for i in range(btch):
        # Forward propagation G
        G = tf.linalg.diag(PHIt[i,0,:])
        for l in range(1,L):
            G = tf.matmul(tf.matmul(tf.linalg.diag(PHIt[i,l,:]),Wl[l,:,:]),G)
        """
        G = tf.linalg.diag(PHIt[i,L-1,:])
        for l in range(L-2,1,-1):
            G = tf.matmul(G,tf.matmul(Wl[l,:,:],tf.linalg.diag(PHIt[i,l-1,:])))
        """
        RR = []
        for k in range(K):  
            
            NU0num0 = Puet[i,k]*tf.abs(tf.matmul(tf.reshape(Ht[k,:,i],(1,N)),tf.matmul(G,np.reshape(W1[:,k],(N,1)))))**2

            NU0den0 = sum([Puet[i,kpr]*tf.abs(tf.matmul(tf.reshape(Ht[k,:,i],(1,N)),tf.matmul(G,np.reshape(W1[:,kpr],(N,1)))))**2 for kpr in range(K) if kpr!= k]) + wp

            #SINR.append(NU0num0/NU0den0)
            
            RR.append(tf.experimental.numpy.log2(1 + NU0num0/NU0den0))
            print(RR)
            #RR[i,k] = tf.experimental.numpy.log2(1 + NU0num0/NU0den0) #tf.concat(tf.concat(tf.concat(R0,R1),R2),R3)

        indices = tf.reshape(i, [1,1])
        updates = tf.reshape(tf.reduce_min(RR,axis=0),[1,1])
        updates = tf.cast(updates, tf.float64)
        sumR = tf.tensor_scatter_nd_update(sumR, indices, updates)
        
    return -tf.reduce_mean(sumR, axis=-1)


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
L = 5 #meatsurfaces
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

###### Neural Network parameter
### Batch size
btch = 1 #32
### Channel realizations
Nch = 1 #30*btch

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
### Genarate channel matrix used for validation
H1,_ = generate_channels(K,N,Nch,Betak,Rnn)
#H1 = generate_channels(K,N,Nch,Betak,Rnn) # For training
#savemat("channel.mat", {"H":H1})
#H2 = generate_channels(K,N,Nch,Betak,Rnn) # For validation
H2=H1
### Initialization
Htrain0 = np.reshape(H1,(K*N,Nch))
Htrain0r = np.real(Htrain0)
Htrain0i = np.imag(Htrain0)
Htrain = np.concatenate((Htrain0r,Htrain0i),axis=0).transpose()

Hval0 = np.reshape(H2,(K*N,Nch))
Hval0r = np.real(Hval0)
Hval0i = np.imag(Hval0)
Hval = np.concatenate((Hval0r,Hval0i),axis=0).transpose()

### Define the model
# Define layers
shared_input = layers.Input(shape=(2*K*N,1))
#shared_layer1 = layers.Dropout(0.2)(shared_input)
shared_layer1 = layers.Conv1D(3, 32, padding='valid', activation='relu', strides=1)(shared_input) 
shared_layer2 = layers.Flatten()(shared_layer1)
#shared_layer1 = layers.Dense(L*N, activation='relu')(shared_input)
#shared_layer2 = layers.Dense(L*N, activation='relu')(shared_layer1)
shared_layer3 = layers.Dense(L*N, activation='relu')(shared_layer2)
shared_layer4 = layers.Dense(L*N, activation='relu')(shared_layer3)
shared_layer5 = layers.Dense(L*N, activation='relu')(shared_layer4)
# Task 1-specific layers
task1_output = layers.Dense(L*N, activation=custom_activation_phase, name='task1_output')(shared_layer2)
# Create the dual-task model
dual_task_model = Model(inputs=shared_input, outputs=task1_output)

sgd = SGD(learning_rate=0.001, momentum=0.8)

# Compile the model with appropriate loss functions
dual_task_model.compile(optimizer='adam', loss=[CustomLoss])
print (dual_task_model.summary())
### Fit the model : returns a history object
# Enable NumPy behavior in TensorFlow
checkpoint_filepath = './checkpoint.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    save_freq ='epoch', # 1 for every batch
    save_best_only=True
)

weights_dict = {}

weight_callback = tf.keras.callbacks.LambdaCallback \
( on_epoch_end=lambda epoch, logs: weights_dict.update({epoch:dual_task_model.get_weights()}))

# The model weights (that are considered the best) are loaded into the
# model.

# Train model
np_config.enable_numpy_behavior()
history = dual_task_model.fit(x=Htrain, y=Htrain, validation_data=(Hval, Hval), epochs=500, batch_size=btch,callbacks=[weight_callback])
#history = dual_task_model.fit(x=Htrain, y=Htrain, validation_data=(Hval, Hval), epochs=500, batch_size=btch,callbacks=[model_checkpoint_callback])

### Plot
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.grid(True,which="both", linestyle='--')
plt.ylabel('Loss') 
plt.xlabel('Epoch')
plt.xlim([0,1000])
plt.ylim([-10,0])
plt.legend(['Train','Val'], loc='upper right')
plt.show()

ind = np.argmin(history.history['loss'])
# Load the previously saved weights
# Add noise to each element of the weights array
#modified_weights = [weight + 0.0001 for weight in weights_dict[ind-1]]
modified_weights = [weight for weight in weights_dict[ind-1]]
savemat("./channel_model_L%d_b%d_N%d.mat"%(L,NbitQ,N), {"H":H1,'weights':weights_dict[ind-1]})
dual_task_model.set_weights(weights_dict[ind-1])
#dual_task_model.load_weights(checkpoint_filepath)
### Evaluation and Prediction
#H3 = generate_channels(K,N,Nch,Betak,Rnn)
H3=H1

Htest0 = np.reshape(H3,(K*N,Nch))
Htest0r = np.real(Htest0)
Htest0i = np.imag(Htest0)
Htest = np.concatenate((Htrain0r,Htrain0i),axis=0).transpose()
Ht = H3 

PHItest = dual_task_model.predict(Htest)
"""
plt.figure()
plt.plot(PHItest[0])
"""
B = np.array([i*Delta_theta for i in range((2**NbitQ))],dtype=np.float32)
thetas = tf.quantization.fake_quant_with_min_max_args(PHItest,0,((2**NbitQ)-1)*Delta_theta,NbitQ)
"""
PHItest = tf.reshape(PHItest,(Nch,L,N))
thetas = np.zeros((Nch,L,N),dtype=np.float32)
for i in range(btch):
    for l in range(L):
        for n in range(N):
            indx = np.argmin(abs(PHItest[i,l,n]-B))
            thetas[i,l,n] = B[indx]
"""
#y_predt = tf.math.exp(Delta_theta*tf.complex(0.0, PHItest//Delta_theta))
y_predt = tf.math.exp(tf.complex(0.0,thetas))#tf.complex(0.0, PHItest)
#p_predt =  tf.complex(PHItest[:,-K:],0.0)
#p_predt =  budget*PHItest[:,-K:]
#p_predt = PHItest[:,-K:]
rho = [Betak[k,0]/sum(Betak[:,0]) for k in range(K)]
p_predt = np.array([[budget/K for _ in range(K)] for _ in range(Nch)])

PHIt = tf.reshape(y_predt,(Nch,L,N))
Puet = np.array([[budget/K for k in range(K)] for _ in range(Nch)],dtype=np.float64)
#Puet = tf.reshape(p_predt,(Nch,K))
RR = []
sumR = tf.zeros((Nch,1),dtype=tf.float64)
for i in range(0,Nch):    
    # Forward propagation G
    G = tf.linalg.diag(PHIt[i,0,:])
    for l in range(1,L):
        G = tf.matmul(tf.matmul(tf.linalg.diag(PHIt[i,l,:]),Wl[l,:,:]),G)
    R = []
    for k in range(K):  
        
        NU0num0 = Puet[i,k]*tf.abs(tf.matmul(tf.reshape(Ht[k,:,i],(1,N)),tf.matmul(G,np.reshape(W1[:,k],(N,1)))))**2

        NU0den0 = sum([Puet[i,kpr]*tf.abs(tf.matmul(tf.reshape(Ht[k,:,i],(1,N)),tf.matmul(G,np.reshape(W1[:,kpr],(N,1)))))**2 for kpr in range(K) if kpr!= k]) + wp

        #SINR.append(NU0num0/NU0den0)
        
        R.append(tf.experimental.numpy.log2(1 + NU0num0/NU0den0)[0,0].numpy())

    RR.append(R)
    indices = tf.reshape(i, [1,1])
    updates = tf.reshape(tf.reduce_sum(RR[i],axis=0),[1,1])
    updates = tf.cast(updates, tf.float64)
    sumR = tf.tensor_scatter_nd_update(sumR, indices, updates)

############################################# Plotting the simulation results
plt.figure()
Nsnap = Nch
SE=[RR[i][k] for i in range(Nch) for k in range(K)]
SE.sort()
plt.plot(SE,np.array(range(K*Nsnap))/float(K*Nsnap),color='r',label="Rate")

plt.title("Plot")
plt.xlabel("SE [bit/s/Hz]")
plt.ylabel("CDF")
#plt.xlim([0,25])
#plt.ylim([0,1])
# Adding legend to recognize the curve according to it's color
plt.legend()
plt.grid()
plt.show()