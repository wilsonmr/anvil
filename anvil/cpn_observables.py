import numpy as np
from scipy.signal import correlate
import torch

from tqdm import tqdm

from reportengine import collect

import matplotlib.pyplot as plt

from math import ceil

from geometry import Geometry2D

def phi_init(N_samples, L, D):
    phi = np.random.uniform(0, np.pi, size=(N_samples, L**2, D-1))
    phi[:,:,-1] = 2*phi[:,:,-1]
    return phi

def x_calc(phi):
    x = np.zeros((phi.shape[0],phi.shape[1],phi.shape[2]+2))
    for i in range(D-1):
        x[:, :, i] = np.prod(np.sin(phi[:,:,:i]), axis=2) * np.cos(phi[:,:,i])
    x[:,:, -2] = np.prod(np.sin(phi), axis=2)
    return x

def z_calc(x):
    return x[:, :, :N] + x[:, :, N:]*1j

class cpn_observables():

    def __init__(self, L, N, N_samples):
        self.L = L
        self.geometry = Geometry2D(L)
        self.N = N
        self.D = 2*N -1
        self.N_samples = N_samples
        self.z = self.z_calc(self.x_calc(self.phi_init()))
        # print(np.sum(np.conj(self.z) *  self.z, axis=2))
        self.P = self.P_calc(self.z)

    def phi_init(self):
        phi = np.random.uniform(0, np.pi, size=(self.N_samples, self.L**2, self.D-1))
        phi[:,:,-1] = 2*phi[:,:,-1]
        return phi

    @staticmethod
    def x_calc(phi):
        x = np.zeros((phi.shape[0],phi.shape[1],phi.shape[2]+2))
        for i in range(D-1):
            x[:, :, i] = np.prod(np.sin(phi[:,:,:i]), axis=2) * np.cos(phi[:,:,i])
        x[:,:, -2] = np.prod(np.sin(phi), axis=2)
        return x

    @staticmethod
    def z_calc(x):
        return x[:, :, :N] + x[:, :, N:]*1j

    def P_calc(self, z):
        P = np.zeros((self.N_samples, self.L**2, self.N, self.N), dtype=complex)
        zbar = np.conj(z)
        for i in range(self.N_samples):
            for k in range(self.L**2):
                P[i, k, :, :] = np.outer(zbar[i, k, :], z[i, k, :])
        return P

    # def G_calc(self, shift=(1, 1), dims=(0,1)):
    #     P = self.P
    #     # P_shift = np.roll(np.roll(P, shift[0], axis=2), shift[1], axis=3)
    #     shifts = self.geometry.get_shift(shift)
    #     P_shift_a = P[:,shifts[0], :, :]
    #     P_shift = P_shift_a[:, shifts[1], :, :]
    #     prod_tr = np.trace(np.matmul(P_shift, P), axis1=2, axis2=3)
    #     P_tr = np.trace(P, axis1=2, axis2=3)
    #     shift_tr = np.trace(P_shift, axis1=2, axis2=3)
    #     mean_prod = np.average(prod_tr, axis=0)
    #     mean_P, mean_shift = np.average(P_tr, axis=0), np.average(shift_tr, axis=0)
    #     print(np.sum(mean_prod) - np.sum(mean_P*mean_shift))
    #     return np.real(np.sum(mean_prod) - np.sum(mean_P*mean_shift))

    def G_calc(self, shift=(1, 1), dims=(0,1)):
        z = self.z
        shifts = self.geometry.get_shift(shift)
        z_shift_a = z[:, shifts[0], :]
        z_shift = z_shift_a[:, shifts[1], :]
        prod = np.mean(np.sum(np.conj(np.conj(z) * z_shift) * (np.conj(z) * z_shift) , axis=2), axis=0) - 1
        return np.sum(prod)


    def G_tilde_calc(self, p):
        G_tilde = []
        for i in range(self.L):
            for k in range(self.L):
                G_tilde.append( np.exp((np.sum(p * np.array([i,k])))*1j) * self.G_calc(shift=(i,k)))
        return (1/L) * np.sum(G_tilde)

    def chi_m_calc(self):
        return self.G_tilde_calc([0,0])

    def xi_sq_calc(self):
        qm = [2*np.pi / self.L, 0]
        qm_abs = qm[0]
        G_t_0 = self.G_tilde_calc(0)
        G_t_q = self.G_tilde_calc(qm)
        print(f"G t 0: {G_t_0}")
        print(f"G t q: {G_t_q}")
        return (1/(4*(np.sin(qm_abs/2))**2)) * (G_t_0 - G_t_q)/G_t_q

    def Q_calc(self):
        P = self.P
        # shift_down = np.roll(P, 1, axis=2)
        shifts = self.geometry.get_shift()
        shift_down = P[:, shifts[0], :, :]
        # shift_left = np.roll(P, 1, axis=3)
        shift_left = P[:, shifts[1], :, :]
        # shift_both = np.roll(shift_down, 1, axis=3)
        shift_both = P[:, shifts[0], :, :][:, shifts[1], :, :]
        term1 = np.log(np.trace(np.matmul(shift_both, np.matmul(shift_down, P)), axis1=2, axis2=3))
        term2 = np.log(np.trace(np.matmul(shift_left, np.matmul(shift_both, P)), axis1=2, axis2=3))
        return np.sum(np.imag(term1 + term2)/(2*np.pi), axis=1) * 1j

    def chi_t_calc(self):
        volume = self.L**2
        return np.mean(self.Q_calc()**2)/volume


L = 8
N = 2
D = 2*N -1
N_samples = 10

lattice = cpn_observables(L, N, N_samples)

G_t_0 = lattice.chi_m_calc()
xi_sq = lattice.xi_sq_calc()
print(f"xi_sq: {xi_sq}")
chi_t = lattice.chi_t_calc()
Q = lattice.Q_calc()


print(lattice.G_calc(shift=(0,0)))
print(lattice.G_tilde_calc(p=(2*np.pi / lattice.L, 0)))
print(f"chi_m: {G_t_0}")
print(f"xi: {np.sqrt(xi_sq)}")
print(f"Q: {Q}")
print(f"chi_t: {chi_t}")
print(f"chi_t * xi^2: {xi_sq * chi_t}")