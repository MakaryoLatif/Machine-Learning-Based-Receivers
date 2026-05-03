import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from scipy.io import savemat
from tensorflow.keras.utils import Sequence

Nu = 3 # number of single antenna users at the transmitter
Nrx = 100 # number of receive antenna
zeta = 20 # spread parameter that controls how sharp the signal power is spread across the receive antennas
user_peakantenna = [20, 50, 85] # Receive antenna  the users are centered upon

Nsymbols = 10 # sequence length
M = 4 # M-ary constellation
EsNodBs = np.arange(5,20,2) # SNR range
SERs = []
num_samples = 25000000 # number of sequences generated

output_file = "MaMIMO_DL_MNu_SER_results.mat"

psp = np.zeros((Nrx, Nu))
pspRX = np.zeros((Nrx, Nu))

# Calculate power space profile for each user
for ux in range(Nu):
    center = user_peakantenna[ux]
    psp[:, ux] = np.exp(-((np.arange(Nrx) - center)**2) / (2 * zeta**2))
    psp[:, ux] /= np.sum(psp[:, ux])

    pspRX[:,ux] = psp[:, ux]

# DPSK encoding of original symbol block
def generate_M_DPSK_sequence(Nsymbols, M, Nu):
    a = np.random.randint(0,M,size = (Nu, Nsymbols)) # array of information symbol
    a[:, 0] = 0 # first symbol is not detected, only a reference
    b = np.cumprod(np.exp((1j * 2 * np.pi * a) / M), axis=1) # array of DPSK-encoded symbols
    return a, b 

# def add_noise(signal,SNR):
#     No = 1/SNR
#     noise = np.sqrt(No/2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
#     return signal + noise

# def correlation_matrix(received_signal):
#     return received_signal[:,np.newaxis] @ (received_signal[np.newaxis,:]).conj()

# upper triangle of correlation matrix Z
def extract_upper_triangle(matrix):
    return matrix[np.triu_indices_from(matrix,k=1)].view(np.float64).reshape(-1)

#print(extract_upper_triangle(np.eye(10) + 1j*np.eye(10)).shape)

# encode a specific symbol from each user into M^Nu class index
def encode_users_symbols(symbols, M, Nu):
    idx = 0
    for i in range(Nu):
        idx += symbols[i] * (M ** (Nu - i - 1))
    return idx

# decode a specific symbol back to its original form for each user
def decode_users_symbols(idx, M, Nu):
    symbols = []
    for _ in range(Nu):
        symbols.append(idx % M)
        idx //= M
    return symbols[::-1]    

# print(encode_users_symbols([1, 0, 2], 4, 3))
# print(decode_users_symbols(18, 4, 3))

# Dataset generation
def generate_dataset(num_samples, Nsymbols, M, SNR_list, Nu, Nrx, psp, pspRX):
    X = []
    Y = [[] for _ in range(Nsymbols - 1)]
    
    for _ in range(num_samples):
        SNR_dB = np.random.choice(SNR_list) # train on a random SNR point
        SNR =10**(SNR_dB / 10)
        
        a, b = generate_M_DPSK_sequence(Nsymbols,M,Nu)

        # Channel formation follows complex Gaussian distribution with 0 mean and variance=1
        H = (np.random.randn(Nrx, Nu) + 1j * np.random.randn(Nrx, Nu)) / np.sqrt(2)
        for u in range(Nu):
            H[:, u] *= np.sqrt(psp[:, u])
            
        # signal formation without noise or interference
        y = H @ b
        
        # noise generation follows complex Gaussian distribution with 0 mean and variance=1/2*SNR
        noise = np.sqrt(1 / (2 * SNR)) * (np.random.randn(Nrx, Nsymbols) + 1j * np.random.randn(Nrx, Nsymbols))
        # noisy signal formation with AWGN
        yn = y + noise

        yn = np.where(yn.real > 0, 1, -1) + 1j * np.where(yn.imag > 0, 1, -1)

        input_sample = []
        for u in range(Nu):
            Z = yn.conj().T @ np.diag(pspRX[:, u]) @ yn # Correlation matrix formation with spatial dependencies from psp
            input_u = extract_upper_triangle(Z)
            input_sample.append(input_u)
            #X.append(input_u)
        X.append(np.concatenate(input_sample)) # concatenating each user features
        
        # Label formation
        for s in range(1, Nsymbols):
            symbols = [a[u,s] for u in range(Nu)]
            class_idx = encode_users_symbols(symbols, M, Nu)
            one_hot = to_categorical(class_idx, num_classes=M**Nu)
            Y[s-1].append(one_hot)    
        
    X = np.array(X)
    
    Y = [np.array(Ys) for Ys in Y]
            
    return X, Y
    
# X, Y = generate_dataset(num_samples, Nsymbols, M, SNR, Nu, Nrx, psp, pspRX) 
# print("Y shape: ", Y[7].shape)
# print("X shape: ", X.shape)
# print("fisrt Symbol array from list: ", Y[0].shape)
# print(Y)
# Y_stacked = np.stack(Y, axis=1)
# print("Y_stacked shape: ", Y_stacked.shape)
# print(Y_stacked)

# class for on-the-fly generation inheriting from Sequence
class MassiveMIMODataset(Sequence):
    def __init__(self, batch_size, steps_per_epoch, Nsymbols, M, SNR_list, Nu, Nrx, psp, pspRX):
        self.batch_size = batch_size
        self.steps = steps_per_epoch
        self.Nsymbols = Nsymbols
        self.M = M
        self.SNR_list = SNR_list
        self.Nu = Nu
        self.Nrx = Nrx
        self.psp = psp
        self.pspRX = pspRX

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        X, Y = generate_dataset(
            num_samples=self.batch_size,
            Nsymbols=self.Nsymbols,
            M=self.M,
            SNR_list=self.SNR_list,
            Nu=self.Nu,
            Nrx=self.Nrx,
            psp=self.psp,
            pspRX=self.pspRX
        )
        Y_dict = {f'symbol_{i}': Y[i] for i in range(len(Y))}
        return X, Y_dict

# Training phase
batch_size = 1000
steps_per_epoch = num_samples // batch_size

train_generator = MassiveMIMODataset(
    batch_size=batch_size,
    steps_per_epoch=steps_per_epoch,
    Nsymbols=Nsymbols,
    M=M,
    SNR_list=EsNodBs,
    Nu=Nu,
    Nrx=Nrx,
    psp=psp,
    pspRX=pspRX
)

dummy_X, _ = train_generator[0]
input_dim = dummy_X.shape[1]
    
input_layer = Input(shape=(input_dim,))

# the same 2 hidden layers used in the first Architecture
x = Dense(1000, activation='relu')(input_layer)
x = Dense(200, activation='relu')(x)

#  each symbol has a dedicated layer at the output for prediction
output_layer = [Dense(M**Nu, activation='softmax', name=f'symbol_{s}')(x) for s in range(Nsymbols-1)]

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate = 0.0001), loss='categorical_crossentropy', metrics=['accuracy']* (Nsymbols-1))
model.summary()
    
model.fit(train_generator, epochs=25, verbose=1)


# Testing phase
for EsNodB in EsNodBs:
    print(f"\n============================")
    print(f"Running test for EsNodB = {EsNodB}")
    print("============================")
    

    Online_test_samples = 1000000
    X_test, Y_test = generate_dataset(Online_test_samples, Nsymbols, M, np.array([EsNodB]), Nu, Nrx, psp, pspRX)

    Y_pred_prob = model.predict(X_test)

    SERs_per_user = [0 for _ in range(Nu)]
    total_symbols = 0

    # SER Calculation
    for s in range(Nsymbols-1):
        original_idx = np.argmax(Y_test[s], axis=1)
        predicted_idx = np.argmax(Y_pred_prob[s], axis=1)

        for i in range(len(original_idx)):
            true_symbols = decode_users_symbols(original_idx[i], M, Nu)
            pred_symbols = decode_users_symbols(predicted_idx[i], M, Nu)
            
            for u in range(Nu):
                if true_symbols[u] != pred_symbols[u]:
                  SERs_per_user[u] += 1
        total_symbols += len(original_idx)            

    SERs_per_user = [err / total_symbols for err in SERs_per_user]

    for u in range(Nu):
        print(f"User {u} SER = {SERs_per_user[u]:.4f}")

    SERs.append(SERs_per_user)

    print("_____________________________")
    print("_____________________________")


SERs = np.array(SERs).T

for u in range(Nu):
    plt.semilogy(EsNodBs, SERs[u], marker='o', label=f'User {u+1}')

plt.xlabel('EsNodB (dB)')
plt.ylabel('Symbol Error Rate')
plt.title('SER vs EsNodB for Multi-User DL Model')
plt.grid(True, which="both", ls="-")
plt.legend()
plt.savefig("MaMIMO_DL_MNu_SER.png")
#plt.show()
savemat(output_file, {'EsNodB': EsNodBs, 'SERs': SERs})
print(f"SER results saved to {output_file}")
