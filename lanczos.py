import numpy as np
from qutip.qobj import Qobj
from qutip import sigmax, sigmay, sigmaz, tensor, qeye, Qobj

# matriz evolutiva unitaria Harper
def FF(i, j, k, n):
    xi1 = 0.0
    xi2 = 0.0
    FF = np.exp(-1j * 2.0 * np.pi * (float(i - 1) + xi1) * (float(j - 1) + xi2) / float(n)) \
         * np.sqrt(1.0 / float(n))
    return FF

def maUH(k, n):
    UU = np.zeros((n, n), dtype=complex)
    MM = np.zeros((n, n), dtype=complex)
    xi1 = 0.5
    xi2 = 0.5

    for j in range(n):
        for i in range(n):
            UU[i, j] = FF(i, j, k, n) * np.exp(1j * float(n) * k
                          * np.cos(-2.0 * np.pi * (float(i - 1) + xi1) / float(n)))
            MM[i, j] = np.conjugate(FF(j, i, k, n)) * np.exp(1j * float(n) * k
                          * np.cos(-2.0 * np.pi * (float(i - 1) + xi1) / float(n)))
    for j in range(n):
        for i in range(n):
            U[i, j] = 0
            for l in range(n):
                U[i, j] = U[i, j] + MM[i, l] * UU[l, j]

    return U

# matriz del standard map
def maUS(k, n):
    UU = np.zeros((n, n), dtype=complex)
    MM = np.zeros((n, n), dtype=complex)
    xi1 = 0.5
    xi2 = 0.5

    for j in range(n):
        for i in range(n):
            UU[i, j] = FF(i, j, k, n) * np.exp(-1j * float(n) * (k/(2*np.pi))
                          * np.cos(-2.0 * np.pi * (float(i - 1) + xi1) / float(n)))
            MM[i, j] = np.conjugate(FF(j, i, k, n)) * np.exp(-1j * np.pi*(i+xi2)**2/float(n)) 
                        #                                      * k
    return np.matmul(MM,UU)

 

def Kcomplex(F,v,tie):
    Uf=F
    tt=[]
    Kc=[]
    for ii in range(tie-1):
        kk=0
        for i in range(len(v)):
            v1=np.matmul(Uf,v[0])
            v2=np.matmul(np.conjugate(v[i]),v1)
            kk=kk+float(i)*np.abs(v2)**2
        Kc.append(kk)
        tt.append(ii+1)
        Uff=np.matmul(F,Uf)
        Uf=Uff
    return tt,Kc                                 

def r_chaometer(ener,plotadjusted):
    ra = np.zeros(len(ener)-2)
    #center = int(0.1*len(ener))
    #delter = int(0.05*len(ener))
    for ti in range(len(ener)-2):
        ra[ti] = (ener[ti+2]-ener[ti+1])/(ener[ti+1]-ener[ti])
        ra[ti] = min(ra[ti],1.0/ra[ti])
#    print(ra)    
    ra = np.mean(ra)
#    print(ra)
    if plotadjusted == True:
        ra = (ra -0.5307) / (0.3863-0.5307)
#        ra = (ra -0.3863) / (-0.3863+0.5307)
#        ra = (ra -0.5976) / (0.3863-0.5976)
    return ra


def h_ising_transverse(N: int, hx: float, hz: float, jx: float, jy: float,
                       jz: float, to_numpy=False):
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sx
        sx_list.append(tensor(op_list))

        op_list[n] = sy
        sy_list.append(tensor(op_list))

        op_list[n] = sz
        sz_list.append(tensor(op_list))

    # construct the hamiltonian
    H = 0

    # energy splitting terms
    for n in range(N):
        H += hz * sz_list[n]

    for n in range(N):
        H += hx * sx_list[n]

    # interaction terms
    for n in range(N - 1):
        H += -jx * sx_list[n] * sx_list[n + 1] 
        H += -jy * sy_list[n] * sy_list[n + 1] 
        H += -jz * sz_list[n] * sz_list[n + 1]  

    if to_numpy:
        return H.full()
    # H = H.unit()
    return H


def FO_state(H: np.ndarray,
             e0: np.ndarray,
             stop: int = None,
             start_from: int = None,
             ):
    """
    Full orthonormalization Lanczos algorithm for states.
    """
    if isinstance(H, Qobj):
        H = H.full()
    if isinstance(e0, Qobj):
        e0 = e0.full()

    stop = stop or H.shape[0]
    #print(f'stop {stop}.')

    def L(psi): return H @ psi

    def prod(x, y): return x.conj().T @ y

    def norm(e): return np.sqrt(np.real(prod(e, e)))

    if start_from:
        a_arr = start_from[0]
        b_arr = start_from[1]
        e_arr = start_from[2]
        n = len(b_arr)
    else:
        a_arr = [prod(e0, L(e0)) / norm(e0)]
        b_arr = [0]
        e_arr = [e0 / norm(e0)]
        n = 1

    bn = np.inf
    while bn > 1e-7:
        An = L(e_arr[n - 1])

        for _ in range(2):
            for en in e_arr:
                An -= en * prod(en, An)

        bn = norm(An)
        Kn = 1 / bn * An
        an = prod(Kn, L(Kn))

        a_arr.append(an)
        b_arr.append(bn)
        e_arr.append(Kn)

        if n >= stop -1: # ensures that there's always 'stop' elements in basis
            break
        n += 1

    a_arr = np.real(a_arr)
    return np.array(a_arr), np.array(b_arr), np.array(e_arr)
