from functools import partial
import sys
import time

import numpy as np
from scipy import integrate
from qutip import expect, qeye, sigmax, sigmay, sigmaz, tensor, basis, Qobj


def sz_total(N: int, to_numpy=False):
    return sj_total(N, j='z', to_numpy=to_numpy)


def sx_total(N: int, to_numpy=False):
    return sj_total(N, j='x', to_numpy=to_numpy)


def sy_total(N: int, to_numpy=False):
    return sj_total(N, j='y', to_numpy=to_numpy)


def sj_total(N: int, j: str, to_numpy=False):
    """
    Total S_j operator from the sum of N spin-1/2 systems
    j: 'x', 'y', 'z'
    """
    si = qeye(2)

    if j == 'x':
        sj = 0.5*sigmax()
    elif j == 'y':
        sj = 0.5*sigmay()
    elif j == 'z':
        sj = 0.5*sigmaz()

    sj_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sj
        sj_list.append(tensor(op_list))

    sj_total = sum(sj_list)
    if to_numpy:
        sj_total = sj_total.full()
    return sj_total


def sxz_alternate(N: int, to_numpy=False):
    si = qeye(2)

    sx = 0.5*sigmax()
    sz = 0.5*sigmaz()

    sxz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        if n % 2 == 0:
            op_list[n] = sz
        else:
            op_list[n] = sx

        sxz_list.append(tensor(op_list))

    sxz_total = sum(sxz_list)
    if to_numpy:
        sxz_total = sxz_total.full()

    return sxz_total


def sxy_alternate(N: int, to_numpy=False):
    si = qeye(2)

    sx = 0.5*sigmax()
    sy = 0.5*sigmay()

    sxz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        if n % 2 == 0:
            op_list[n] = sy
        else:
            op_list[n] = sx

        sxz_list.append(tensor(op_list))

    sxz_total = sum(sxz_list)
    if to_numpy:
        sxz_total = sxz_total.full()

    return sxz_total


def szy_alternate(N: int, to_numpy=False):
    si = qeye(2)

    sz = 0.5*sigmaz()
    sy = 0.5*sigmay()

    sxz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        if n % 2 == 0:
            op_list[n] = sy
        else:
            op_list[n] = sz

        sxz_list.append(tensor(op_list))

    sxz_total = sum(sxz_list)
    if to_numpy:
        sxz_total = sxz_total.full()

    return sxz_total


def exchange_operator(dim: int, i: int, j: int):
    si = qeye(2)
    geye = [si for k in range(dim)]

    H = 0 * tensor(geye)

    for s in [sigmax(), sigmay(), sigmaz()]:
        g = geye.copy()
        g[i] = s
        g[j] = s
        H += tensor(g)

    H += tensor(geye)
    return H / 2


def parity_operator(n: int):
    si = qeye(2)
    geye = [si for k in range(n)]

    p = tensor(geye)
    for m in range(n // 2):
        p = exchange_operator(n, m, n - m - 1) * p
    return p


def parity_bra(psi, n=None):
    """return <P|psi>"""
    if n is None:
        n = int(np.log2(psi.shape[0]))
    p = parity_operator(n)
    return p * psi


def to_even_state(psi):
    n = int(np.log2(psi.shape[0]))
    p = parity_operator(n)
    return ((p + 1) * psi).unit()


class QOperator:
    def __init__(self, op):
        self.op = Qobj(op)
        self.n = int(np.log2(op.shape[0]))
        self.transformations = []

    def to_numpy(self):
        return self.op.full()

    def to_Qobj(self):
        return self.op

    def to_sz_subspace(self, eigenval=0, aprox_symmetry=False):
        transformation = partial(self._to_subspace, symmetry=sz_total(self.n),
                                 eigenval=eigenval, aprox_symmetry=aprox_symmetry)

        self.op = transformation(Op=self.op)
        self.transformations.append(transformation)
        return self

    def to_even_subspace(self, eigenval=1, aprox_symmetry=False):
        transformation = partial(self._to_subspace,
                                 symmetry=parity_operator(self.n),
                                 eigenval=eigenval, aprox_symmetry=aprox_symmetry)

        self.op = transformation(Op=self.op)
        self.transformations.append(transformation)
        return self

    def _to_subspace(self, Op, symmetry, eigenval, aprox_symmetry,
                     apply_prev_hamiltonian_transformations=True):

        if apply_prev_hamiltonian_transformations:  # this avoids recursion
            for t in self.transformations:
                symmetry = t(symmetry,
                             apply_prev_hamiltonian_transformations=False)

        if aprox_symmetry: # WIP
            Op_evals, Op_evecs = Op.eigenstates()
            s_arr = np.real([(v.dag()*symmetry*v).full()[0][0]
                             for v in Op_evecs])

            subspace, = np.where(np.round(s_arr, 1) == eigenval)

            Op_sub = Op.transform(Op_evecs).extract_states(subspace)
            # print(Op_evals[subspace])
            # Op_evecs_sub = np.column_stack([v[subspace]
                                            # for v in Op_evecs[subspace]])
            # Op_evals_sub = np.diag(Op_evals[subspace])

            # Op_sub = Op_evecs_sub.conj().T @ Op_evals_sub @ Op_evecs_sub
            # Op_sub = Qobj(Op_sub)
            # print(np.round(Op_sub.eigenenergies(), 10))
        else:
            s_evals, s_evecs = symmetry.eigenstates()
            Op_trans = Op.transform(s_evecs)
            subspace, = np.where(s_evals == eigenval)
            Op_sub = Op_trans.extract_states(subspace)

        return Op_sub


def to_even_subspace(H):
    if isinstance(H, np.ndarray):
        is_numpy = True
        H = Qobj(H)
    else:
        is_numpy = False

    n = int(np.log2(H.shape[0]))
    P = parity_operator(n)

    P_evals, P_evecs = P.eigenstates()

    H_par = H.transform(P_evecs)

    subspace, = np.where(np.isclose(P_evals, 1))
    H_sub = H_par.extract_states(subspace)

    return H_sub

    # if isinstance(H, np.ndarray):
    # is_numpy = True
    # H = Qobj(H)
    # else:
    # is_numpy = False

    # n = int(np.log2(H.shape[0]))
    # p = parity_operator(n)

    # p_evals, p_evecs = p.eigenstates()

    # n_mone, n_one = np.bincount(p_evals.astype(int))  # -1 and 1 ocurrencies
    # p_evecs_matrix = np.column_stack([vec.full() for vec in p_evecs])

    # H_par = H.transform(np.linalg.inv(p_evecs_matrix))
    # H_sub = H_par.extract_states(range(n_mone, n_one + n_mone))

    # if is_numpy:
    # H_sub = H_sub.full()

    return H_sub


def to_sz_subspace(H, m=1):
    """m: float. sz proyection"""

    if isinstance(H, np.ndarray):
        is_numpy = True
        H = Qobj(H)
    else:
        is_numpy = False

    n = int(np.log2(H.shape[0]))
    sz = sz_total(n)

    s_evals, s_evecs = sz.eigenstates()

    p_evecs_matrix = np.column_stack([vec.full() for vec in s_evecs])
    H_spin = H.transform(np.linalg.inv(p_evecs_matrix))

    s_evals = np.round(s_evals, 10)
    # ^ numeric errors in the last digits can be a problem
    subspace = np.where(s_evals == m)
    i1 = subspace[0][0]
    i2 = subspace[0][-1]
    H_sub = H_spin.extract_states(range(i1, i2+1))

    if is_numpy:
        H_sub = H_sub.full()

    return H_sub


def is_even_state(psi):
    return np.all(psi == parity_bra(psi))


def odeintz(func, z0, t, **kwargs):
    """An odeint-like function for complex valued differential equations."""

    # Disallow Jacobian-related arguments.
    _unsupported_odeint_args = ['Dfun', 'col_deriv', 'ml', 'mu']
    bad_args = [arg for arg in kwargs if arg in _unsupported_odeint_args]
    if len(bad_args) > 0:
        raise ValueError("The odeint argument %r is not supported by "
                         "odeintz." % (bad_args[0],))

    # Make sure z0 is a numpy array of type np.complex128.
    z0 = np.array(z0, dtype=np.complex128, ndmin=1)

    def realfunc(x, t, *args):
        z = x.view(np.complex128)
        dzdt = func(z, t, *args)
        # func might return a python list, so convert its return
        # value to an array with type np.complex128, and then return
        # a np.float64 view of that array.
        return np.asarray(dzdt, dtype=np.complex128).view(np.float64)

    result = integrate.odeint(realfunc, z0.view(np.float64), t, **kwargs)

    if kwargs.get('full_output', False):
        z = result[0].view(np.complex128)
        infodict = result[1]
        return z, infodict
    else:
        z = result.view(np.complex128)
        return z


def wavefunction_phi(a, b, tmax, ntimes):
    """
    Output
    ------
    t.shape = (ntimes,)
    phi.shape = (ntimes, b.size)
    """

    def eq_syst(phi, t):
        dot_phi = np.zeros_like(b, dtype=complex)

#        print(b[0], b[1])
#        print(a[0], a[1])

        dot_phi[0] = -b[1] * phi[1] + 1j*a[0]*phi[0]
        dot_phi[1:-1] = b[1:-1] * phi[:-2] - b[2:] * phi[2:] + 1j*a[1:-1]*phi[1:-1]
        dot_phi[-1] = b[-1] * phi[-2] + 1j*a[-1]*phi[-1]

#        dot_phi[0] = -b[1] * phi[1]
#        dot_phi[1:-1] = b[1:-1] * phi[:-2] - b[2:] * phi[2:]
#        dot_phi[-1] = b[-1] * phi[-2]

        return dot_phi

    k = b.size

    phi0 = [1+0j] + [0+0j] * (k - 1)
    # t = np.linspace(0, np.exp(2 * N) * 4, ntimes)
    t = np.linspace(0, tmax, ntimes)
     # PARA GASTÓN: Esto de np.exp(N) * 4 es ridículo.
     # El tiempo este debería ser ajustable e independiente de N a este nivel
    phi = odeintz(eq_syst, phi0, t)

    return t, phi


def k_complexity_from_phi(phi):
#    kc = np.sum(np.arange(phi.shape[1]) * phi**2, axis=1)
    kc = np.sum(np.arange(phi.shape[1]) * np.abs(phi)**2, axis=1)
    return kc


def k_complexity(a,b, tmax, ntimes=500):
    t, phi = wavefunction_phi(a, b, tmax, ntimes)
    kc = k_complexity_from_phi(phi)
    return t, kc


def computational_to_physic_base(psi):
    b0 = basis(2, 0)
    b1 = basis(2, 1)

    probabilities = []
    for qb in range(int(np.log2(psi.shape[0]))):
        ptrace = psi.ptrace(qb).unit()

        pb0, pb1 = expect(ptrace, b0), expect(ptrace, b1)
        probabilities.append([pb0, pb1])

    probabilities = np.array(probabilities)

    return probabilities


def computational_base_to_fstr(psi, unicode=True, simplify=False):
    probabilities = computational_to_physic_base(psi)

    if unicode:
        down, up, mixed = '↓', '↑', '⇅'
    else:
        down, up, mixed = 'd', 'u', 'm'
    s_list = []
    for probs in probabilities:
        b0p, b1p = probs
        if np.isclose(b0p, 1):
            s = down
        elif np.isclose(b1p, 1):
            s = up
        else:
            if simplify:
                s = mixed
            else:
                s = f'({b0p:.2g}{down} + {b1p:.2g}{up})'
        s_list.append(s)

    out = ' '.join(s_list)

    return out


def chaometer(H: 'hamiltonian'):
    IWD = 0.5307
    IP = 0.3863

    e = np.sort(H.eigenenergies())
    s = np.diff(e)
    r = s[:-1] / s[1:]
    r_tilde = np.min(np.column_stack([r, 1 / r]), axis=1)
    eta = (np.mean(r_tilde) - IP) / (IWD - IP)
    return np.real(eta)


def chaometer_from_energies(e_arr: 'enegies'):
    IWD = 0.5307
    IP = 0.3863

    e = np.sort(e_arr)
    s = np.diff(e)
    r = s[:-1] / s[1:]
    r_tilde = np.min(np.column_stack([r, 1 / r]), axis=1)
    eta = (np.mean(r_tilde) - IP) / (IWD - IP)
    return np.real(eta)


def __test_parity_operators():
    psi = tensor(
        basis(2, 0),
        basis(2, 1),
        basis(2, 1),
    )

    # Parity things
    print(f'\n{psi = }')
    print(f'\n{parity_bra(psi) = }')
    print(f'\n{to_even_state(psi) = }')


def __test_computational_base_to_fstr():
    down = basis(2, 0)
    up = basis(2, 1)
    psi = tensor((np.sqrt(1 / 4) * down + np.sqrt(3 / 4) * up).unit(), up, up)

    print(f'\n{computational_base_to_fstr(psi) = }')
    print(f'\n{computational_base_to_fstr(psi, simplify=True) = }')
