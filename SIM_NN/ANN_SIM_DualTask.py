###### Importing modules
from functions import *

def NormP_Q(sig):
    # Phase shift matrix 
    ypredQ =  tf.quantization.fake_quant_with_min_max_args(sig,0,((2**NbitQ)-1),NbitQ)
    print(ypredQ)
    return ypredQ

def CustomLoss(y_true,y_pred): 
    #Ht = tf.convert_to_tensor(H, dtype=tf.complex64)
    print("Shape y_true: ",y_true.shape)
    print("Shape H: ",H.shape)
    print("y_pred_power: ",y_pred[:,-K:])
    # Retrieve Htrain0r and Htrain0i
    num_rows = y_true.shape[1] // 2
    Htrain0r_retrieved = Htrain[:,:num_rows].transpose()
    Htrain0i_retrieved = Htrain[:,:num_rows].transpose()
    Hri = Htrain0r_retrieved + 1j* Htrain0i_retrieved
    #print("Shape Hri: ",Hri.shape)
    Ht = tf.reshape(Hri,(K,N,btch))
    y_predt = tf.math.exp(tf.complex(0.0,Delta_theta*(y_pred[:,:L*N]//Delta_theta)))
    PHIt = tf.reshape(y_predt,(btch,L,N))
    Puet = tf.reshape(budget*y_pred[:,-K:],(btch,K))
    sumR = tf.zeros((btch,1),dtype=float)
    minR = tf.zeros((btch,1),dtype=float)
    Rmin = tf.constant([0.00005])
    #print("Rmin: ",Rmin)

    for i in range(btch):
        # Forward propagation G
        G = tf.linalg.diag(PHIt[i,0,:])
        for l in range(1,L):
            G = tf.matmul(tf.matmul(tf.linalg.diag(PHIt[i,l,:]),Wl[l,:,:]),G)

        RR = []
        for k in range(K):  
            
            NU0num0 = Puet[i,k]*tf.abs(tf.matmul(tf.reshape(Ht[k,:,i],(1,N)),tf.matmul(G,np.reshape(W1[:,k],(N,1)))))**2

            NU0den0 = sum([Puet[i,kpr]*tf.abs(tf.matmul(tf.reshape(Ht[k,:,i],(1,N)),tf.matmul(G,np.reshape(W1[:,kpr],(N,1)))))**2 for kpr in range(K) if kpr!= k]) #+ wp
            #tf.print("Gain : ",NU0num0)
            #tf.print("Interference : ",NU0den0)
            #SINR.append(NU0num0/NU0den0)
            RR.append(tf.experimental.numpy.log2(1 + NU0num0/NU0den0))
            #RR.append(tf.math.maximum(Rmin,tf.experimental.numpy.log2(1 + NU0num0/NU0den0)))
            #RR[i,k] = tf.experimental.numpy.log2(1 + NU0num0/NU0den0) #tf.concat(tf.concat(tf.concat(R0,R1),R2),R3)

        indices = tf.reshape(i, [1,1])
        # Sum Rate
        updates = tf.reshape(tf.reduce_sum(RR),[1,1])
        sumR = tf.tensor_scatter_nd_update(sumR, indices, updates)
        # # Min Rate
        # updates = tf.reshape(tf.reduce_min(RR,axis=0),[1,1])
        # minR = tf.tensor_scatter_nd_update(minR, indices, updates)

    return -tf.reduce_mean(sumR, axis=0) #-tf.reduce_mean(minR, axis=0)

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
K_factor = 10
### SIM
L = 5 #meatsurfaces
Nx = 7
Ny = 7
N = Nx*Ny #meta-atoms
NbitQ = 2
c = 3e8
f = 28e9 #Hz
B = 10e6 #Hz
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
btch = 1
### Channel realizations
Nch = 1

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
H,_ = generate_channels(K,N,Nch,Betak,Rnn) # For training
H2 = H # For validation

### Initialization
Htrain0 = np.reshape(H,(K*N,btch))
Htrain0r = np.real(Htrain0)
Htrain0i = np.imag(Htrain0)
Htrain = np.concatenate((Htrain0r,Htrain0i),axis=0).transpose()
print("Shape Htrain: ",Htrain.shape)
print("Shape Htrain0r: ",Htrain0r.shape)
print("Shape Htrain0i: ",Htrain0i.shape)

Hval0 = np.reshape(H2,(K*N,btch))
Hval0r = np.real(Hval0)
Hval0i = np.imag(Hval0)
Hval = np.concatenate((Hval0r,Hval0i),axis=0).transpose()
### Define the model
# Shared layers
shared_input = layers.Input(shape=(2*K*N,))
shared_layer1 = layers.Dense(392, activation='relu')(shared_input)
shared_layer2 = layers.Dense(392, activation='relu')(shared_layer1)
shared_layer3 = layers.Dense(392, activation='relu')(shared_layer2)
shared_layer4 = layers.Dense(392, activation='relu')(shared_layer3)
# Task 1-specific layers
task1_output = layers.Dense(L*N, activation=custom_activation_phase, name='task1_output')(shared_layer2)
#task1_output = Lambda(lambda x: NormP_Q(x))(PHIn)
# Task 2-specific layers
task2_output = layers.Dense(K, activation='softmax', name='task2_output')(shared_layer2)

# Create the dual-task model
dual_task_model = Model(inputs=shared_input, outputs=tf.concat([task1_output,task2_output],axis=-1))

# Compile the model with appropriate loss functions
dual_task_model.compile(optimizer='adam', loss=[CustomLoss])
print(dual_task_model.summary())

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


### Fit the model : returns a history object
history = dual_task_model.fit(x=Htrain, y=Htrain, validation_data=(Hval, Hval), epochs=50, batch_size=btch,callbacks=[weight_callback])
"""
history = dual_task_model.fit(
    Htrain,
    Htrain,
    validation_data=(Hval, Hval),
    batch_size = btch,
    epochs = 300,
    shuffle=True)
"""
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

### Evaluation and Prediction
# Load the previously saved weights
ind = np.argmin(history.history['loss'])
# Add noise to each element of the weights array
#modified_weights = [weight + 0.0001 for weight in weights_dict[ind-1]]
modified_weights = [weight for weight in weights_dict[ind-1]]

dual_task_model.set_weights(weights_dict[ind-1])

H3 = H

Htest0 = np.reshape(H3,(K*N,Nch))
Htest0r = np.real(Htest0)
Htest0i = np.imag(Htest0)
Htest = np.concatenate((Htest0r,Htest0i),axis=0).transpose()

PHItest = dual_task_model.predict(Htest)

Ht = tf.convert_to_tensor(H, dtype=tf.complex64)
y_predt = tf.math.exp(tf.complex(0.0, Delta_theta*(PHItest[:,:L*N]//Delta_theta)))
#y_predt = tf.math.exp((2*np.pi/2**NbitQ)*tf.complex(0.0, PHItest[:,:L*N]))
#p_predt =  tf.complex(PHItest[:,-K:],0.0)
p_predt =  budget*PHItest[:,-K:]
#p_predt = PHItest[:,-K:]
PHIt = tf.reshape(y_predt,(Nch,L,N))
Puet = tf.reshape(p_predt,(Nch,K))
scaled_pk = np.zeros((Nch,K),dtype=float)

sumR = tf.zeros((Nch,1),dtype=float)
for i in range(0,Nch):    
    # Forward propagation G
    G = tf.linalg.diag(PHIt[i,0,:])
    for l in range(1,L):
        G = tf.matmul(tf.matmul(tf.linalg.diag(PHIt[i,l,:]),Wl[l,:,:]),G)
    RR = []
    for k in range(K):  
        
        scaled_pk[i,k] = (Puet[i,k]/sum(Puet[i,:]))*budget

        NU0num0 = Puet[i,k]*tf.abs(tf.matmul(tf.reshape(Ht[k,:,i],(1,N)),tf.matmul(G,np.reshape(W1[:,k],(N,1)))))**2

        NU0den0 = sum([Puet[i,kpr]*tf.abs(tf.matmul(tf.reshape(Ht[k,:,i],(1,N)),tf.matmul(G,np.reshape(W1[:,kpr],(N,1)))))**2 for kpr in range(K) if kpr!= k]) + wp

        #SINR.append(NU0num0/NU0den0)
        
        RR.append(tf.experimental.numpy.log2(1 + NU0num0/NU0den0))

    indices = tf.reshape(i, [1,1])
    updates = tf.reshape(tf.reduce_sum(RR,axis=0),[1,1])
    sumR = tf.tensor_scatter_nd_update(sumR, indices, updates)
