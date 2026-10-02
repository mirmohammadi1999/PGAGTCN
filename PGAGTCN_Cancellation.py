import numpy as np
import fullduplex_hybrid as fd
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Conv1D, Add, Multiply, Lambda
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt
import os
import warnings
import random
from scipy.signal import savgol_filter
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def dc_aware_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    y_true_c = tf.complex(y_true[:, 0], y_true[:, 1])
    y_pred_c = tf.complex(y_pred[:, 0], y_pred[:, 1])
    dc_loss = tf.square(tf.abs(tf.reduce_mean(y_true_c - y_pred_c)))
    return mse_loss + 0.05 * dc_loss

def get_tdnn_features(x_sig, memory_len, chanLen):
    n_samples = x_sig.size - chanLen
    features = np.zeros((n_samples, 2 * memory_len))
    pad_len = max(0, memory_len - chanLen)
    x_padded = np.pad(x_sig, (pad_len, 0), mode='constant')

    for i in range(n_samples):
        end_idx = i + chanLen + pad_len
        start_idx = end_idx - memory_len
        taps = x_padded[start_idx : end_idx]
        taps = np.flip(taps)
        features[i, :memory_len] = taps.real
        features[i, memory_len:] = taps.imag
    return features

def get_nn_target(y_sig, chanLen):
    return np.hstack((np.reshape(y_sig[chanLen:].real, (-1, 1)), np.reshape(y_sig[chanLen:].imag, (-1, 1))))

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ---------------------------------------------------------
# (PSD , Loss)
# ---------------------------------------------------------
def plot_psd(y_orig, y_lin_canc, y_nl_canc, noise, params, dataset_name, filename):
    samplingFreqMHz = params['samplingFreqMHz']
    fftpoints = 4096
    scalingConst = samplingFreqMHz * 1e6
    freqAxis = np.linspace(-samplingFreqMHz/2, samplingFreqMHz/2, fftpoints)
    savgolWindow = 45
    savgolDegree = 1

    def compute_psd(signal_array):
        fft_sig = np.fft.fftshift(np.fft.fft(signal_array / np.sqrt(scalingConst), fftpoints, axis=0, norm="ortho"))
        return 10 * np.log10(savgol_filter(np.power(np.abs(fft_sig), 2), savgolWindow, savgolDegree))

    psd_orig = compute_psd(y_orig)
    psd_lin = compute_psd(y_lin_canc)
    psd_nl = compute_psd(y_nl_canc)
    psd_noise = compute_psd(noise)

    power_orig = 10 * np.log10(np.mean(np.abs(y_orig)**2))
    power_lin = 10 * np.log10(np.mean(np.abs(y_lin_canc)**2))
    power_nl = 10 * np.log10(np.mean(np.abs(y_nl_canc)**2))
    power_noise = 10 * np.log10(np.mean(np.abs(noise)**2))

    target_orig_dbm = -42.7 if samplingFreqMHz == 20 else -43.7
    offset_db = target_orig_dbm - power_orig

    plt.figure(figsize=(9, 6.5), dpi=150)
    plt.plot(freqAxis, psd_orig + offset_db, 'b-', label=f'SI Signal ({power_orig + offset_db:.1f} dBm)', linewidth=2)
    plt.plot(freqAxis, psd_lin + offset_db, 'r-', label=f'Linear DC ({power_lin + offset_db:.1f} dBm)', linewidth=2)
    plt.plot(freqAxis, psd_nl + offset_db, 'm-', label=f'PG-AGTCN DC ({power_nl + offset_db:.1f} dBm)', linewidth=2)
    plt.plot(freqAxis, psd_noise + offset_db, 'k:', label=f'Noise Floor ({power_noise + offset_db:.1f} dBm)', linewidth=2.5)

    plt.xlabel('Frequency (MHz)', fontsize=13, fontweight='bold')
    plt.ylabel('Power Spectral Density (dBm/Hz)', fontsize=13, fontweight='bold')
    plt.xlim([-samplingFreqMHz/2, samplingFreqMHz/2])

    if samplingFreqMHz == 20:
        plt.ylim([-190, -110])
    else:
        plt.ylim([-190, -110])

    plt.grid(which='both', linestyle='--', alpha=0.5)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=11, framealpha=1.0, edgecolor='black')

    plt.savefig(f'figures/{filename}.svg', format='svg', bbox_inches='tight')
    plt.close()

def plot_loss(history, dataset_name, filename):
    plt.figure(figsize=(7, 5), dpi=150)
    epochs = range(1, len(history.history['loss']) + 1)

    plt.plot(epochs, history.history['loss'], 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r--', linewidth=2, label='Validation Loss')

    plt.title(f'Learning Curve - {dataset_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('DC-Aware Loss (Log Scale)', fontsize=12)
    plt.yscale('log')
    plt.grid(linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(f'figures/{filename}.svg', format='svg')
    plt.close()

# ---------------------------------------------------------
def build_model_tcn_only(tdnn_mem):
    inp = Input(shape=(tdnn_mem, 2))
    c1 = Conv1D(16, 3, padding='causal', dilation_rate=1, activation='relu')(inp)
    c2 = Conv1D(16, 3, padding='causal', dilation_rate=2, activation='relu')(c1)
    c3 = Conv1D(16, 3, padding='causal', dilation_rate=4, activation='relu')(c2)
    skip = Conv1D(16, 1, padding='same')(inp)
    res_out = Add()([c3, skip])
    last_step = Lambda(lambda x: x[:, -1, :])(res_out)
    out = Dense(2, activation='linear')(last_step)
    return Model(inp, out, name="Sparse_TCN")

def build_model_gate_only(tdnn_mem):
    inp = Input(shape=(tdnn_mem, 2))
    last_step = Lambda(lambda x: x[:, -1, :])(inp)
    d1 = Dense(16, activation='relu')(last_step)
    gate = Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-4), bias_initializer=tf.keras.initializers.Constant(-1.0))(d1)
    nn_res = Dense(2, activation='linear')(d1)
    out = Multiply()([nn_res, gate])
    return Model(inp, out, name="Sparse_Gating")

def build_model_pg_agtcn(tdnn_mem):
    inp = Input(shape=(tdnn_mem, 2))
    c1 = Conv1D(16, 3, padding='causal', dilation_rate=1, activation='relu')(inp)
    c2 = Conv1D(16, 3, padding='causal', dilation_rate=2, activation='relu')(c1)
    c3 = Conv1D(16, 3, padding='causal', dilation_rate=4, activation='relu')(c2)
    skip = Conv1D(16, 1, padding='same')(inp)
    res_out = Add()([c3, skip])
    last_step = Lambda(lambda x: x[:, -1, :])(res_out)
    gate = Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-4), bias_initializer=tf.keras.initializers.Constant(-1.0))(last_step)
    nn_res = Dense(2, activation='linear')(last_step)
    out = Multiply()([nn_res, gate])
    return Model(inp, out, name="PG_AGTCN")

# ---------------------------------------------------------
datasets_info = [
    {
        'name': 'Dataset 1 (10dBm, 20MHz)',
        'path': 'data/fdTestbedData20MHz10dBm.mat',
        'params': {
            'samplingFreqMHz': 20, 'hSILen': 13, 'pamaxordercanc': 7, 'dataOffset': 14,
            'nEpochs': 60, 'learningRate': 0.002, 'batchSize': 32, 'tdnn_mem': 15, 'lasso_alpha': 1e-4
        }
    },
    {
        'name': 'Dataset 2 (32dBm, 80MHz)',
        'path': 'data/signal-20MHz_sampled-80MHz_txAmp-8_circulator_passive-50dB_iq-0.mat',
        'params': {
            'samplingFreqMHz': 80, 'hSILen': 13, 'pamaxordercanc': 7, 'dataOffset': 12,
            'nEpochs': 60, 'learningRate': 0.002, 'batchSize': 32, 'tdnn_mem': 15, 'lasso_alpha': 5e-4
        }
    }
]

data_sizes = [2000, 3000, 4000, 5000]
seeds = [42, 1000, 2026]

if not os.path.exists('figures'):
    os.makedirs('figures')

data_efficiency_results = {ds['name']: {'sizes': data_sizes, 'Sparse': [], 'Proposed': []} for ds in datasets_info}

for ds in datasets_info:
    print("\n" + "="*90)
    print(f"STARING COMPREHENSIVE REVIEWER-PROOF BENCHMARK: {ds['name']}")
    print("="*90)

    params = ds['params']
    chanLen = params['hSILen']
    tdnn_mem = params['tdnn_mem']

    x_full, y_full, noise_full, _ = fd.loadData(ds['path'], params)

    # ------------------------------------------------------------------
    # PART 1: Data Efficiency Loop
    # ------------------------------------------------------------------
    print(f"\n[PHASE 1] Executing Data-Efficiency Analysis...")
    set_all_seeds(42)

    for N_total in data_sizes:
        tf.keras.backend.clear_session()
        x = x_full[:N_total]
        y = y_full[:N_total]
        N_valid = N_total - chanLen
        N_train = int(N_valid * 0.8)
        N_val   = int(N_valid * 0.1)

        x_train, y_train = x[0 : N_train + chanLen], y[0 : N_train + chanLen]
        val_start = N_train
        x_val, y_val = x[val_start : val_start + N_val + chanLen], y[val_start : val_start + N_val + chanLen]
        test_start = val_start + N_val
        x_test, y_test = x[test_start :], y[test_start :]

        y_train_tgt = y_train[chanLen:]
        y_test_eval = y_test[chanLen:]
        power_orig  = np.mean(np.abs(y_test_eval)**2)

        nBasis = int((params['pamaxordercanc']+1)/2*( (params['pamaxordercanc']+1)/2 +1))
        def build_A(x_sig):
            A_mat = np.zeros((x_sig.size-chanLen, nBasis*chanLen), dtype=np.complex128)
            matInd = 0
            for i in range(1, params['pamaxordercanc']+1, 2):
                for j in range(0, i+1):
                    xnl = np.power(x_sig, j) * np.power(np.conj(x_sig), i-j)
                    A_mat[:, matInd*chanLen:(matInd+1)*chanLen] = np.reshape([np.flip(xnl[k+1:k+chanLen+1], axis=0) for k in range(xnl.size-chanLen)], (xnl.size-chanLen, chanLen))
                    matInd += 1
            return A_mat

        A_train, A_val, A_test = build_A(x_train), build_A(x_val), build_A(x_test)
        A_real = np.vstack((np.hstack((A_train.real, -A_train.imag)), np.hstack((A_train.imag, A_train.real))))
        y_real = np.hstack((y_train_tgt.real, y_train_tgt.imag))

        scaler = StandardScaler()
        A_scaled = scaler.fit_transform(A_real)
        lasso = Lasso(alpha=params['lasso_alpha'], fit_intercept=False, max_iter=10000).fit(A_scaled, y_real)

        support = np.abs(lasso.coef_) > 1e-6
        c_deb = np.zeros_like(lasso.coef_)
        if np.sum(support) > 0:
            c_deb[support] = np.linalg.lstsq(A_real[:, support], y_real, rcond=None)[0]

        n_f = A_train.shape[1]
        h_sp = c_deb[:n_f] + 1j * c_deb[n_f:]

        yC_sp_tr = (A_train @ h_sp)
        yC_sp_v  = (A_val @ h_sp)
        yC_sp_t  = (A_test @ h_sp)

        sic_sparse = 10 * np.log10(power_orig / np.mean(np.abs(y_test_eval - yC_sp_t)**2))
        data_efficiency_results[ds['name']]['Sparse'].append(sic_sparse)

        X_tr_cnn = np.stack((get_tdnn_features(x_train, tdnn_mem, chanLen)[:, :tdnn_mem], get_tdnn_features(x_train, tdnn_mem, chanLen)[:, tdnn_mem:]), axis=-1)
        X_v_cnn  = np.stack((get_tdnn_features(x_val, tdnn_mem, chanLen)[:, :tdnn_mem], get_tdnn_features(x_val, tdnn_mem, chanLen)[:, tdnn_mem:]), axis=-1)
        X_t_cnn  = np.stack((get_tdnn_features(x_test, tdnn_mem, chanLen)[:, :tdnn_mem], get_tdnn_features(x_test, tdnn_mem, chanLen)[:, tdnn_mem:]), axis=-1)

        res_tr = y_train_tgt - yC_sp_tr
        res_v  = y_val[chanLen:] - yC_sp_v
        Y_tr = get_nn_target(np.pad(res_tr, (chanLen, 0)), chanLen)
        Y_v  = get_nn_target(np.pad(res_v, (chanLen, 0)), chanLen)
        y_var = np.var(Y_tr)
        Y_tr, Y_v = Y_tr / np.sqrt(y_var), Y_v / np.sqrt(y_var)

        model = build_model_pg_agtcn(tdnn_mem)
        model.compile(loss=dc_aware_loss, optimizer=Adam(learning_rate=params['learningRate']))
        model.fit(X_tr_cnn, Y_tr, validation_data=(X_v_cnn, Y_v), epochs=params['nEpochs'], batch_size=params['batchSize'], verbose=0)

        pred = model.predict(X_t_cnn, verbose=0)
        yC_nl = np.squeeze(pred[:, 0] + 1j*pred[:, 1]) * np.sqrt(y_var)
        sic_prop = 10 * np.log10(power_orig / np.mean(np.abs((y_test_eval - yC_sp_t) - yC_nl)**2))
        data_efficiency_results[ds['name']]['Proposed'].append(sic_prop)
        print(f"   [Size: {N_total}] Sparse: {sic_sparse:.1f} dB | Proposed: {sic_prop:.1f} dB")

    # ------------------------------------------------------------------
    # PART 2: Statistical Ablation, PSD Plotting & Loss Plotting (N=5000)
    # ------------------------------------------------------------------
    print(f"\n[PHASE 2] Executing Multi-Seed Ablation Study, PSD & Loss Plots (N=5000) ...")
    N_total = 5000
    x = x_full[:N_total]
    y = y_full[:N_total]
    N_valid = N_total - chanLen
    N_train = int(N_valid * 0.8)
    N_val   = int(N_valid * 0.1)

    x_train, y_train = x[0 : N_train + chanLen], y[0 : N_train + chanLen]
    val_start = N_train
    x_val, y_val = x[val_start : val_start + N_val + chanLen], y[val_start : val_start + N_val + chanLen]
    test_start = val_start + N_val
    x_test, y_test = x[test_start :], y[test_start :]

    noise_test_eval = noise_full[test_start:test_start + x_test.size][chanLen:]

    y_train_tgt = y_train[chanLen:]
    y_test_eval = y_test[chanLen:]
    power_orig  = np.mean(np.abs(y_test_eval)**2)

    # --- Linear Estimation ---
    A_lin_train = np.reshape([np.flip(x_train[i+1:i+chanLen+1], axis=0) for i in range(x_train.size-chanLen)], (x_train.size-chanLen, chanLen))
    h_lin = np.linalg.lstsq(A_lin_train, y_train_tgt, rcond=None)[0]
    A_lin_test = np.reshape([np.flip(x_test[i+1:i+chanLen+1], axis=0) for i in range(x_test.size-chanLen)], (x_test.size-chanLen, chanLen))
    y_lin_pred = A_lin_test @ h_lin
    y_lin_canc = y_test_eval - y_lin_pred
    # ----------------------------------------------

    A_train, A_val, A_test = build_A(x_train), build_A(x_val), build_A(x_test)
    A_real = np.vstack((np.hstack((A_train.real, -A_train.imag)), np.hstack((A_train.imag, A_train.real))))
    y_real = np.hstack((y_train_tgt.real, y_train_tgt.imag))

    scaler = StandardScaler()
    A_scaled = scaler.fit_transform(A_real)
    lasso = Lasso(alpha=params['lasso_alpha'], fit_intercept=False, max_iter=20000).fit(A_scaled, y_real)

    support = np.abs(lasso.coef_) > 1e-6
    c_deb = np.zeros_like(lasso.coef_)
    if np.sum(support) > 0:
        c_deb[support] = np.linalg.lstsq(A_real[:, support], y_real, rcond=None)[0]
    n_f = A_train.shape[1]
    h_sp = c_deb[:n_f] + 1j * c_deb[n_f:]

    yC_sp_tr = (A_train @ h_sp)
    yC_sp_v  = (A_val @ h_sp)
    yC_sp_t  = (A_test @ h_sp)
    sic_sparse_base = 10 * np.log10(power_orig / np.mean(np.abs(y_test_eval - yC_sp_t)**2))

    X_tr_cnn = np.stack((get_tdnn_features(x_train, tdnn_mem, chanLen)[:, :tdnn_mem], get_tdnn_features(x_train, tdnn_mem, chanLen)[:, tdnn_mem:]), axis=-1)
    X_v_cnn  = np.stack((get_tdnn_features(x_val, tdnn_mem, chanLen)[:, :tdnn_mem], get_tdnn_features(x_val, tdnn_mem, chanLen)[:, tdnn_mem:]), axis=-1)
    X_t_cnn  = np.stack((get_tdnn_features(x_test, tdnn_mem, chanLen)[:, :tdnn_mem], get_tdnn_features(x_test, tdnn_mem, chanLen)[:, tdnn_mem:]), axis=-1)

    res_tr = y_train_tgt - yC_sp_tr
    res_v  = y_val[chanLen:] - yC_sp_v
    Y_tr = get_nn_target(np.pad(res_tr, (chanLen, 0)), chanLen)
    Y_v  = get_nn_target(np.pad(res_v, (chanLen, 0)), chanLen)
    y_var = np.var(Y_tr)
    Y_tr, Y_v = Y_tr / np.sqrt(y_var), Y_v / np.sqrt(y_var)

    results_pgagtcn = []
    best_y_nl_canc = None
    best_history = None

    for s in seeds:
        tf.keras.backend.clear_session()
        set_all_seeds(s)

        # PG-AGTCN
        m_pg = build_model_pg_agtcn(tdnn_mem)
        m_pg.compile(loss=dc_aware_loss, optimizer=Adam(learning_rate=params['learningRate']))
        history_pg = m_pg.fit(X_tr_cnn, Y_tr, validation_data=(X_v_cnn, Y_v), epochs=params['nEpochs'], batch_size=params['batchSize'], verbose=0)
        p_pg = m_pg.predict(X_t_cnn, verbose=0)
        y_pg = np.squeeze(p_pg[:, 0] + 1j*p_pg[:, 1]) * np.sqrt(y_var)

        y_nl_canc = (y_test_eval - yC_sp_t) - y_pg
        sic_val = 10 * np.log10(power_orig / np.mean(np.abs(y_nl_canc)**2))
        results_pgagtcn.append(sic_val)

        best_y_nl_canc = y_nl_canc
        best_history = history_pg
        params_pg = m_pg.count_params()

    print(f"\n[INFO] Saving PSD & Loss Plots for {ds['name']}...")
    filename_psd = 'PSD_Dataset_1' if params['samplingFreqMHz'] == 20 else 'PSD_Dataset_2'
    filename_loss = 'LossCurve_Dataset_1' if params['samplingFreqMHz'] == 20 else 'LossCurve_Dataset_2'

    plot_psd(y_test_eval, y_lin_canc, best_y_nl_canc, noise_test_eval, params, ds['name'], filename_psd)
    plot_loss(best_history, ds['name'], filename_loss)


# ---------------------------------------------------------
# Data Efficiency
# ---------------------------------------------------------
print("\n[INFO] Generating Separated Data Efficiency Plots...")

# Dataset 1
fig1, ax1 = plt.subplots(figsize=(7, 5), dpi=150)
ax1.plot(data_sizes, data_efficiency_results[datasets_info[0]['name']]['Sparse'], 's--', color='#5CB85C', linewidth=2, label='Sparse Polynomial')
ax1.plot(data_sizes, data_efficiency_results[datasets_info[0]['name']]['Proposed'], 'o-', color='#5BC0DE', linewidth=2.5, label='Proposed PG-AGTCN')
ax1.set_title('Dataset 1 (10dBm)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Training Samples', fontsize=12)
ax1.set_ylabel('Total SIC (dB)', fontsize=12)
ax1.grid(linestyle='--', alpha=0.6)
ax1.legend(fontsize=11)
fig1.tight_layout()
fig1.savefig('figures/DataEfficiency_Dataset1.svg', format='svg')
plt.close(fig1)

# Dataset 2
fig2, ax2 = plt.subplots(figsize=(7, 5), dpi=150)
ax2.plot(data_sizes, data_efficiency_results[datasets_info[1]['name']]['Sparse'], 's--', color='#5CB85C', linewidth=2, label='Sparse Polynomial')
ax2.plot(data_sizes, data_efficiency_results[datasets_info[1]['name']]['Proposed'], 'o-', color='#5BC0DE', linewidth=2.5, label='Proposed PG-AGTCN')
ax2.set_title('Dataset 2 (32dBm)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Training Samples', fontsize=12)
ax2.set_ylabel('Total SIC (dB)', fontsize=12)
ax2.grid(linestyle='--', alpha=0.6)
ax2.legend(fontsize=11)
fig2.tight_layout()
fig2.savefig('figures/DataEfficiency_Dataset2.svg', format='svg')
plt.close(fig2)

print("[SUCCESS] Separated figures generated: 'DataEfficiency_Dataset1.svg' & 'DataEfficiency_Dataset2.svg'.")

print("\n" + "="*85)
print(f"{'COMPUTATIONAL COMPLEXITY & FLOPs ANALYSIS (For Table III/IV)':^85}")
print("="*85)
print("The FLOPs are calculated based on inference phase complexity.")
print("PG-AGTCN TCN layers (Causal 1D Conv) FLOPs per step:")
print(" - C_in = 2, C_out = 16, K = 3  => 2 * 2 * 16 * 3 = 192")
print(" - C_in = 16, C_out = 16, K = 3 => 2 * 16 * 16 * 3 = 1536")
print(" - C_in = 16, C_out = 16, K = 3 => 2 * 16 * 16 * 3 = 1536")
print(" - Skip Conv (1x1)              => 2 * 2 * 16 * 1 = 64")
print(" - Dense Gate & Output          => (2*16*1)+(2*16*2) = 96")
print("Total NN inference FLOPs per sample: ~ 3.4k FLOPs.")