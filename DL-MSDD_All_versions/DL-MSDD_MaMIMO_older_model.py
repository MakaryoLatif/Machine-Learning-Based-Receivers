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

Nu = 1 # number of single antenna users at the transmitter
Nrx = 100 # number of receive antenna
zeta = 20 # spread parameter that controls how sharp the signal power is spread across the receive antennas
user_peakantenna = [50] # Receive antenna  the users are centered upon

Nsymbols = 10 # sequence length
M = 4 # M-ary constellation
EsNodBs = np.arange(5,20,2) # SNR range
SERs = []
num_samples = 5000000 # number of sequences generated

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

# encode the whole symbol block for each user into M^(Nsymbols-1) class index
def encode_users_symbols(symbols, M, Nu):
    idx = 0
    for i in range(Nu):
        idx += symbols[i] * (M ** (Nu - i - 1))
    return idx

# decode the whole symbol block back to its original form for each user
def decode_users_symbols(idx, M, Nu):
    symbols = []
    for _ in range(Nu):
        symbols.append(idx % M)
        idx //= M
    return symbols[::-1]    

# print(encode_users_symbols([1, 0, 2], 4, 3))
# print(decode_users_symbols(18, 4, 3))

# Dataset Generation
def generate_dataset(num_samples, Nsymbols, M, SNR_list, Nu, Nrx, psp, pspRX):
    X = []
    Y = []
    
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

        #yn = np.where(yn.real > 0, 1, -1) + 1j * np.where(yn.imag > 0, 1, -1)

        input_sample = []
        for u in range(Nu):
            Z = yn.conj().T @ np.diag(pspRX[:, u]) @ yn # Correlation matrix formation with spatial dependencies from psp
            input_u = extract_upper_triangle(Z)
            input_sample.append(input_u)
            #X.append(input_u)
        X.append(np.concatenate(input_sample)) # concatenating each user features
        
        # Label formation
        symbols_block = []
        for s in range(1, Nsymbols):
            for u in range(Nu):
                symbols_block.append(a[u, s])
        class_idx = encode_users_symbols(symbols_block, M, Nu*(Nsymbols-1))
        one_hot = to_categorical(class_idx, num_classes=M**(Nu*(Nsymbols-1)))
        Y.append(one_hot)
        
    X = np.array(X)
    
    Y = np.array(Y)
            
    return X, Y
    
# X, Y = generate_dataset(num_samples, Nsymbols, M, SNR, Nu, Nrx, psp, pspRX) 
# print("Y shape: ", Y[7].shape)
# print("X shape: ", X.shape)
# print("fisrt Symbol array from list: ", Y[0].shape)
# print(Y)
# Y_stacked = np.stack(Y, axis=1)
# print("Y_stacked shape: ", Y_stacked.shape)
# print(Y_stacked)

# Training phase
X, Y = generate_dataset(num_samples, Nsymbols, M, EsNodBs, Nu, Nrx, psp, pspRX)
    
input_layer = Input(shape=(X.shape[1],))

# the same 2 hidden layers used in the first Architecture
x = Dense(1000, activation='relu')(input_layer)
x = Dense(200, activation='relu')(x)

# One layer with M^(Nsymbols-1) neurons responsible for predicting the whole symbol block
output_layer = Dense(M**(Nu*(Nsymbols-1)), activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate = 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
    
model.fit(X, Y, epochs=10, batch_size=32, verbose=1)

# Testing phase
for EsNodB in EsNodBs:
    print(f"\n============================")
    print(f"Running test for EsNodB = {EsNodB}")
    print("============================")
    

    Online_test_samples = 1000000
    X_test, Y_test = generate_dataset(Online_test_samples, Nsymbols, M, np.array([EsNodB]), Nu, Nrx, psp, pspRX)

    Y_pred_prob = model.predict(X_test)
    original_idx = np.argmax(Y_test, axis=1)
    predicted_idx = np.argmax(Y_pred_prob, axis=1)

    SERs_per_user = [0 for _ in range(Nu)]
    total_symbols = 0


    # SER Calculations
    for i in range(len(original_idx)):
        true_symbols = decode_users_symbols(original_idx[i], M, Nu * (Nsymbols - 1))
        pred_symbols = decode_users_symbols(predicted_idx[i], M, Nu * (Nsymbols - 1))
        for s in range(Nsymbols-1):
            for u in range(Nu):
                idx = s * Nu + u
                if true_symbols[idx] != pred_symbols[idx]:
                  SERs_per_user[u] += 1
        total_symbols += (Nsymbols-1)*Nu

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
plt.title('SER vs EsNodB for Single-User DL Model')
plt.grid(True, which="both", ls="-")
plt.legend()
plt.savefig("MaMIMO_DL_MNu_SER_single_user_low_res(1).png")
#plt.show()
savemat(output_file, {'EsNodB': EsNodBs, 'SERs': SERs})
print(f"SER results saved to {output_file}")
