from qutip import basis, rand_ket, tensor

from functions_d import to_even_state


def all_down_one_up(N, n, to_numpy=False):
    """Estado de N-1 spins |0> y uno |1> en el lugar n"""
    psi_list = [basis(2, 0)] * N
    psi_list[n] = basis(2, 1)

    psi0 = tensor(psi_list)
    psi0 = psi0 / psi0.norm()
    if to_numpy:
        psi0 = psi0.full()
    return psi0


def all_down_one_symmetric(N, n, psi, to_numpy=False):
    """Estado de N-1 spins |0> y uno |psi> en el lugar n"""
    psi_list = [basis(2, 0)] * N
    psi_list[n] = psi
    psi_list[~n] = psi

    psi0 = tensor(psi_list)
    psi0 = psi0 / psi0.norm()
    if to_numpy:
        psi0 = psi0.full()
    return psi0


def all_down_one_symmetric_up(N, n, to_numpy=False):
    """Estado de N-1 spins |0> y uno |1> en el lugar n y N-n"""
    psi0 = all_down_one_symmetric(N, n, basis(2, 1), to_numpy=False)
    if to_numpy:
        psi0 = psi0.full()
    return psi0


def all_random(N, to_numpy=False):
    """
    Estado de N spins random del estilo (a|0> + (1-a)|1>)/sqrt(2)
    """
    psi0 = rand_ket(2**N, dims=[[2] * N, [1] * N])
    if to_numpy:
        psi0 = psi0.full()
    return psi0


def all_random_even(N, to_numpy=False):
    """
    Estado de N spins random del estilo (a|0> + (1-a)|1>)/sqrt(2)
    """
    psi0 = rand_ket(2**N, dims=[[2] * N, [1] * N])
    if to_numpy:
        psi0 = psi0.full()
    return to_even_state(psi0)


def random_spin(to_numpy=False):
    psi = rand_ket(2)
    if to_numpy:
        psi = psi.full()
    return psi
