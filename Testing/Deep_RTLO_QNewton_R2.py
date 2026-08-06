import numpy as np
from numba import jit
import math
import ast
import os


@jit(nopython=True, fastmath=True)
def Activation(x, mode_code=0):
    '''mode_code: 0=tanh, 1=sigmoid, 2=relu'''
    if mode_code == 0:
        return np.tanh(x)
    elif mode_code == 1:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        return np.maximum(0.0, x)

@jit(nopython=True, fastmath=True)
def dActivation(x, mode_code=0):
    '''mode_code: 0=tanh, 1=sigmoid, 2=relu'''
    if mode_code == 0:
        return 1.0 - np.tanh(x)**2
    elif mode_code == 1:
        s = 1.0 / (1.0 + np.exp(-x))
        return s * (1.0 - s)
    else:
        res = np.zeros_like(x)
        for i in range(x.size):
            if x.flat[i] > 0:
                res.flat[i] = 1.0
        return res

@jit(nopython=True, fastmath=True)
def _lbfgs_two_loop_kernel(g_k, s_hist, y_hist, k_mem):
    q = g_k.copy()
    alphas = np.zeros(k_mem)

    # Loop 1: Direção Regressiva
    for i in range(k_mem - 1, -1, -1):
        s_i = s_hist[i]
        y_i = y_hist[i]
        rho_i = 1.0 / (np.dot(y_i, s_i) + 1e-10)
        alpha_i = rho_i * np.dot(s_i, q)
        alphas[i] = alpha_i
        q -= alpha_i * y_i

    if k_mem > 0:
        s_last = s_hist[-1]
        y_last = y_hist[-1]
        gamma_k = np.dot(s_last, y_last) / (np.dot(y_last, y_last) + 1e-10)
    else:
        gamma_k = 1.0

    r = gamma_k * q

    # Loop 2: Direção Progressiva
    for i in range(k_mem):
        s_i = s_hist[i]
        y_i = y_hist[i]
        rho_i = 1.0 / (np.dot(y_i, s_i) + 1e-10)
        beta = rho_i * np.dot(y_i, r)
        r += s_i * (alphas[i] - beta)

    return -r

@jit(nopython=True, fastmath=True)
def _fit_kernel(xP_b, yR, wO, wI_tuple, wR_tuple, hP_tuple, tau, act_code):
    n_layers = len(wI_tuple)
    curr_in = xP_b
    layer_inputs = [curr_in]
    
    new_uP = []
    new_hP = []

    # Forward Pass
    for l in range(n_layers):
        u = wR_tuple[l] @ hP_tuple[l] + wI_tuple[l] @ curr_in
        h = hP_tuple[l] * (1.0 - 1.0 / tau) + Activation(u, act_code) / tau
        new_uP.append(u)
        new_hP.append(h)
        
        curr_in = np.empty(h.shape[0] + 1)
        curr_in[:-1] = h
        curr_in[-1] = 1.0
        layer_inputs.append(curr_in)

    yP = wO @ new_hP[-1]
    eS = yR - yP

    # Backward Pass
    g_O = -np.outer(eS, new_hP[-1])
    grad_wI = [np.empty((1, 1))] * n_layers
    grad_wR = [np.empty((1, 1))] * n_layers

    dh_next = -(wO.T @ eS)

    for l in range(n_layers - 1, -1, -1):
        du = (dh_next / tau) * dActivation(new_uP[l], act_code)
        grad_wR[l] = np.outer(du, hP_tuple[l])
        grad_wI[l] = np.outer(du, layer_inputs[l])

        if l > 0:
            dh_next = wI_tuple[l][:, :-1].T @ du

    grad_flat_size = g_O.size
    for l in range(n_layers):
        grad_flat_size += grad_wI[l].size + grad_wR[l].size

    g_k = np.empty(grad_flat_size)
    idx = 0
    g_k[idx:idx + g_O.size] = g_O.flatten()
    idx += g_O.size

    for l in range(n_layers):
        sz_i = grad_wI[l].size
        g_k[idx:idx + sz_i] = grad_wI[l].flatten()
        idx += sz_i

        sz_r = grad_wR[l].size
        g_k[idx:idx + sz_r] = grad_wR[l].flatten()
        idx += sz_r

    return g_k, new_uP, new_hP

@jit(nopython=True, fastmath=True)
def _predict_intr_kernel(xP, xL, xU, ep, wO, wR_list, wI_list, hP2_list, hL2_list, hU2_list, tau, act_code, flw_past, n_idx):
    xP = np.append(xP, 1.0)
    xL = np.append(xL, 1.0)
    xU = np.append(xU, 1.0)
    
    if not flw_past:
        n_idx = 0

    curr_inP = xP
    curr_inL = xL
    curr_inU = xU

    wOU = np.maximum((1.0 + ep) * wO, wO / (1.0 + ep))
    wOL = np.minimum((1.0 + ep) * wO, wO / (1.0 + ep))

    n_layers = len(wR_list)
    new_hP2 = []
    new_hL2 = []
    new_hU2 = []

    for l in range(n_layers):
        wR, wI = wR_list[l], wI_list[l]
        hP, hL, hU = hP2_list[l].copy(), hL2_list[l].copy(), hU2_list[l].copy()

        wRU = np.maximum((1.0 + ep) * wR, wR / (1.0 + ep))
        wRL = np.minimum((1.0 + ep) * wR, wR / (1.0 + ep))
        wIU = np.maximum((1.0 + ep) * wI, wI / (1.0 + ep))
        wIL = np.minimum((1.0 + ep) * wI, wI / (1.0 + ep))

        uP = (wR @ hP) + (wI @ curr_inP)
        uL = (wRL @ hL) + (wIL @ curr_inL)
        uU = (wRU @ hU) + (wIU @ curr_inU)

        uU_tmp = np.maximum(uU, uL)
        uL_tmp = np.minimum(uU, uL)
        uU, uL = uU_tmp, uL_tmp

        hP = hP * (1.0 - 1.0 / tau) + Activation(uP, act_code) / tau
        hL = hL * (1.0 - 1.0 / tau) + Activation(uL, act_code) / tau
        hU = hU * (1.0 - 1.0 / tau) + Activation(uU, act_code) / tau

        hU_tmp = np.maximum(hU, hL)
        hL_tmp = np.minimum(hU, hL)
        hU, hL = hU_tmp, hL_tmp

        new_hP2.append(hP)
        new_hL2.append(hL)
        new_hU2.append(hU)

        curr_inP = np.append(hP, 1.0)
        curr_inL = np.append(hL, 1.0)
        curr_inU = np.append(hU, 1.0)

    yP = (wO @ new_hP2[-1])
    yL = (wOL @ new_hL2[-1])
    yU = (wOU @ new_hU2[-1])

    yU_tmp = np.maximum(yU, yL)
    yL_tmp = np.minimum(yU, yL)
    yU, yL = yU_tmp, yL_tmp

    return yL[n_idx], yP[n_idx], yU[n_idx], new_hP2, new_hL2, new_hU2

@jit(nopython=True, fastmath=True)
def _predict_single_kernel(xP, wO, wR_list, wI_list, hP2_list, tau, act_code, flw_past, n_idx):
    xP = np.append(xP, 1.0)
    if not flw_past:
        n_idx = 0
        
    curr_in = xP
    new_hP2 = []
    
    for l in range(len(wR_list)):
        hP = hP2_list[l].copy()
        uP = (wR_list[l] @ hP) + (wI_list[l] @ curr_in)
        hP = hP * (1.0 - 1.0 / tau) + Activation(uP, act_code) / tau
        new_hP2.append(hP)
        curr_in = np.append(hP, 1.0)

    yP = (wO @ new_hP2[-1])
    return yP[n_idx], new_hP2

# =====================================================================
# CLASSE RTLO
# =====================================================================

def XavierUniform(shape, sd):
    np.random.seed(sd)
    n_in, n_out = shape
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, size=shape)

class RTLO:
    def __init__(self, nI, nR, nO, ηS=[0.1], τ=10, mode='past', act='tanh', m_history=10):
        np.random.seed(42)
        if isinstance(nR, (int, np.integer)):
            self.nR = [nR]
        else:
            self.nR = list(nR)
            
        self.n_layers = len(self.nR)
        self.k = 1
        self.j = nI - 1
        self.t = np.array([])
        self.start = 0
        self.ref = None
        self.act = act
        self.act_code = {'tanh': 0, 'sigmoid': 1, 'relu': 2}[act]
        self.flw = mode
        self.n = -1
        self.nI, self.nO = nI + 1, nO
        self.ηS = np.array(ηS)
        self.τ = float(τ)
        self.ρ = 0.003
        self.x = np.zeros(nI + 1)
        
        self.hP = [0.1 * np.ones(size) for size in self.nR]
        self.hU = [0.1 * np.ones(size) for size in self.nR]
        self.hL = [0.1 * np.ones(size) for size in self.nR]
        self.hP2 = [0.1 * np.ones(size) for size in self.nR]
        self.hU2 = [0.1 * np.ones(size) for size in self.nR]
        self.hL2 = [0.1 * np.ones(size) for size in self.nR]
        self.uP = [np.zeros(size) for size in self.nR]
        self.uP2 = [np.zeros(size) for size in self.nR]

        self.wI = [XavierUniform([self.nR[0], nI + 1], sd=42)]
        for l in range(1, self.n_layers):
            self.wI.append(XavierUniform([self.nR[l], self.nR[l-1] + 1], sd=42 + l))

        self.wR = [XavierUniform([self.nR[l], self.nR[l]], sd=41 + l) for l in range(self.n_layers)]
        self.wO = XavierUniform([nO, self.nR[-1]], sd=40)

        self.yP, self.yR, self.yL, self.yU = [np.array([]) for i in range(4)]
        self.yP_hist = np.zeros(self.nI)
        self.εY, self.εM, self.εR, self.εE, self.eP, self.eR, self.ΣW = [0 for i in range(7)]
        self.εM_hist, self.εR_hist, self.eR_hist, self.eP_hist = [np.array([]) for i in range(4)]
        self.wR_hist = [[] for _ in range(self.n_layers)]
        self.wI_hist = [[] for _ in range(self.n_layers)]
        self.wO_hist = []

        self.m_history = m_history
        self.s_history = []
        self.y_history = []
        self.g_prev = None
        self.w_prev = None

    def _pack_params(self):
        params = [self.wO.flatten()]
        for l in range(self.n_layers):
            params.append(self.wI[l].flatten())
            params.append(self.wR[l].flatten())
        return np.concatenate(params)

    def _unpack_params(self, w_vec):
        idx = 0
        shape_wO, size_wO = self.wO.shape, self.wO.size
        self.wO = w_vec[idx:idx + size_wO].reshape(shape_wO)
        idx += size_wO

        for l in range(self.n_layers):
            shape_wI, size_wI = self.wI[l].shape, self.wI[l].size
            self.wI[l] = w_vec[idx:idx + size_wI].reshape(shape_wI)
            idx += size_wI

            shape_wR, size_wR = self.wR[l].shape, self.wR[l].size
            self.wR[l] = w_vec[idx:idx + size_wR].reshape(shape_wR)
            idx += size_wR

    def fit(self, xP, yR):
        if self.flw != 'past': self.n = 0
        xP_b = np.append(xP, 1.0)

        g_k, new_uP, new_hP = _fit_kernel(
            xP_b, yR, self.wO,
            tuple(self.wI), tuple(self.wR), tuple(self.hP),
            self.τ, self.act_code
        )

        w_k = self._pack_params()

        if self.g_prev is not None and self.w_prev is not None:
            s_k = w_k - self.w_prev
            y_k = g_k - self.g_prev

            if np.dot(y_k, s_k) > 1e-8:
                if len(self.s_history) >= self.m_history:
                    self.s_history.pop(0)
                    self.y_history.pop(0)
                self.s_history.append(s_k)
                self.y_history.append(y_k)

        k_mem = len(self.s_history)
        if k_mem > 0:
            s_arr = np.array(self.s_history)
            y_arr = np.array(self.y_history)
        else:
            s_arr = np.empty((0, w_k.size))
            y_arr = np.empty((0, w_k.size))

        d_k = _lbfgs_two_loop_kernel(g_k, s_arr, y_arr, k_mem)

        η = self.ηS[0]
        w_new = w_k + η * d_k

        self.w_prev = w_k.copy()
        self.g_prev = g_k.copy()

        self._unpack_params(w_new)

        for l in range(self.n_layers):
            self.wR_hist[l].append(self.wR[l].flatten())
            self.wI_hist[l].append(self.wI[l].flatten())
        self.wO_hist.append(self.wO.flatten())

        self.hP = [h.copy() for h in new_hP]
        self.hP2 = [h.copy() for h in new_hP]
        self.hL = [h.copy() for h in new_hP]
        self.hU = [h.copy() for h in new_hP]
        self.uP = new_uP
        self.x = xP_b
        self.k += 1

    def PredictSingle(self, xP, show=False):
        yP, new_hP2 = _predict_single_kernel(
            xP, self.wO, tuple(self.wR), tuple(self.wI), tuple(self.hP2),
            self.τ, self.act_code, (self.flw == 'past'), self.n
        )
        self.hP2 = list(new_hP2)
        return yP

    def PredictIntr(self, xP, xL, xU, ep, show=False):
        yL, yP, yU, new_hP2, new_hL2, new_hU2 = _predict_intr_kernel(
            xP, xL, xU, float(ep), self.wO,
            tuple(self.wR), tuple(self.wI),
            tuple(self.hP2), tuple(self.hL2), tuple(self.hU2),
            self.τ, self.act_code, (self.flw == 'past'), self.n
        )
        self.hP2 = list(new_hP2)
        self.hL2 = list(new_hL2)
        self.hU2 = list(new_hU2)

        if show:
            print('-----------------')

        return np.array([yL, yP, yU])

    def Restore(self):
        self.hP2 = [h.copy() for h in self.hP]
        self.hL2 = [h.copy() for h in self.hP]
        self.hU2 = [h.copy() for h in self.hP]

    def ReturnParameters(self):
        return [self.wR, self.wI, self.wO, self.hP, self.hP2, self.hL2, self.hU2, self.x]

    def ReceiveParameters(self, vec):
        self.wR, self.wI, self.wO, self.hP, self.hP2, self.hL2, self.hU2, self.x = vec