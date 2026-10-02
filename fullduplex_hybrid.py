from sklearn.linear_model import Lasso
import numpy as np
import scipy.io
import sys

def loadData(fileName, params):
    dataOffset = params['dataOffset']
    chanLen = params['hSILen']
    offset = np.maximum(dataOffset-int(np.ceil(chanLen/2)),1)
    matFile = scipy.io.loadmat(fileName)
    
    x = np.squeeze(matFile['txSamples'], axis=1)[:-offset]
    y = np.squeeze(matFile['analogResidual'], axis=1)[offset:]
    y = y - np.mean(y)
    
    # --- (RMS Normalization) ---
    rms_val = np.sqrt(np.mean(np.abs(x)**2))
    x = x / rms_val
    y = y / rms_val
    # -------------------------------------------------
    
    if 'noiseSamples' in matFile:
        noise = np.squeeze(matFile['noiseSamples'], axis=1)
        noise = noise / rms_val
        noisePower = np.squeeze(matFile['noisePower'], axis=1)
    else:
        noise = np.zeros_like(y)
        noisePower = 0.0
        
    return x, y, noise, noisePower

def SIestimationLinear(x, y, params):
    chanLen = params['hSILen']
    A = np.reshape([np.flip(x[i+1:i+chanLen+1],axis=0) for i in range(x.size-chanLen)], (x.size-chanLen, chanLen))
    h = np.linalg.lstsq(A, y[chanLen:], rcond=None)[0]
    return h

def SIestimationNonLinear(x, y, params):
    print("Self-interference channel estimation:")
    pamaxorder = params.get('pamaxordercanc', 1)
    chanLen = params['hSILen']
    nBasisFunctions = int(( pamaxorder+1)/2*( (pamaxorder+1)/2 +1))
    A = np.zeros((x.size-chanLen, nBasisFunctions*chanLen), dtype=np.complex128)

    matInd = 0
    for i in range(1,pamaxorder+1,2):
        for j in range(0,i+1):
            sys.stdout.write("\r1. Constructing basis functions... ({:d}/{:d})".format(int(matInd+1),nBasisFunctions))
            sys.stdout.flush()
            xnl = np.power(x,j)*np.power(np.conj(x),i-j)
            A[:,matInd*chanLen:(matInd+1)*chanLen] = np.reshape([np.flip(xnl[i+1:i+chanLen+1],axis=0) for i in range(xnl.size-chanLen)], (xnl.size-chanLen, chanLen))
            matInd += 1

    sys.stdout.write("\r1. Constructing basis functions... done!         \n")
    sys.stdout.write("2. Doing channel estimation with Lasso... ")
    sys.stdout.flush()
    
    A_real = np.vstack((np.hstack((A.real, -A.imag)), np.hstack((A.imag, A.real))))
    y_real = np.hstack((y[chanLen:].real, y[chanLen:].imag))

    # الگوریتم Lasso بدون نیاز به اسکیلرهای مخرب پایتون
    lasso = Lasso(alpha=1e-5, fit_intercept=False, max_iter=5000, tol=1e-4)
    lasso.fit(A_real, y_real)
    
    c_real = lasso.coef_
    n_features = A.shape[1]
    h = c_real[:n_features] + 1j * c_real[n_features:]
    
    active_params = np.sum(np.abs(h) > 1e-6) * 2 
    
    print(f"\n=====================================================")
    print(f"[SPARSITY REPORT]: Out of {h.size*2} total parameters,")
    print(f"Lasso kept ONLY {active_params} active parameters!")
    print(f"Complexity reduced by: {100 - (active_params/(h.size*2)*100):.1f}%")
    print(f"=====================================================\n")

    return h

def SIcancellationLinear(x, h, params):
    xcan = np.convolve(x, h, mode='full')
    return xcan[0:x.size]

def SIcancellationNonLinear(x, h, params):
    pamaxorder = params['pamaxordercanc']
    chanLen = params['hSILen']
    nBasisFunctions = int(( pamaxorder+1)/2*( (pamaxorder+1)/2 +1))
    xcan = np.zeros(x.size+chanLen-1, dtype=np.complex128)

    chanInd = 0
    for i in range(1,pamaxorder+1,2):
        for j in range(0,i+1):
            sys.stdout.write("\r1. Constructing basis functions and cancellation signal... ({:d}/{:d})".format(int(chanInd+1),int(nBasisFunctions)))
            sys.stdout.flush()
            xnl = np.power(x,j)*np.power(np.conj(x),i-j)
            xcan += np.convolve(xnl, h[chanInd*chanLen:(chanInd+1)*chanLen])
            chanInd += 1

    return xcan[0:x.size]