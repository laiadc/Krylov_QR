from qiskit.opflow import StateFn
from qiskit.opflow import Z,X,Y
from qiskit.quantum_info import Pauli
from qiskit.opflow import *
from qiskit.circuit.library import Diagonal
from qiskit.extensions import  UnitaryGate
from qiskit import QuantumCircuit
from qiskit import BasicAer
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector
from scipy.stats import unitary_group
from qiskit.quantum_info import pauli_basis
import random
import numpy as np
import itertools

class QuantumCircQiskit:
    def __init__(self, gates_name, num_gates=50,nqbits=8,observables_type = 'single'):
        
        self.num_gates = num_gates
        self.gates_name = gates_name
        self.observables_type = observables_type
        self.gates_set = []
        self.qubits_set = []
        self.nqbits=nqbits
        if self.gates_name=='G1':
            gates = ['CNOT', 'H', 'X']
        if self.gates_name=='G2':
            gates = ['CNOT', 'H', 'S']
        if self.gates_name=='G3':
            gates = ['CNOT', 'H', 'T']  
        
        qubit_idx = list(range(self.nqbits))
        # Store gates
        if self.gates_name in ['G1', 'G2', 'G3']:
            for i in range(self.num_gates):
                # Select random gate
                gate = random.sample(gates,1)[0] 
                self.gates_set.append(gate)
                if gate=='CNOT':
                    # Select qubit 1 and 2 (different qubits)
                    qbit1 = random.sample(qubit_idx,1)[0]
                    qubit_idx2 = qubit_idx.copy()
                    qubit_idx2.remove(qbit1)
                    qbit2 = random.sample(qubit_idx2,1)[0]
                    self.qubits_set.append([qbit1, qbit2])
                else:
                    # Select qubit
                    qbit = random.sample(qubit_idx,1)[0]
                    self.qubits_set.append([qbit])
        elif self.gates_name=='D2':
            qubit_idx = list(range(self.nqbits))
            self.qubits_set = list(itertools.combinations(qubit_idx, 2))
            self.phis = np.random.uniform(0, 2*np.pi, size=(len(self.qubits_set), 2**2))
        elif self.gates_name=='D3':
            qubit_idx = list(range(self.nqbits))
            self.qubits_set = list(itertools.combinations(qubit_idx, 3))
            self.phis = np.random.uniform(0, 2*np.pi, size=(len(self.qubits_set), 2**3))
        elif self.gates_name=='Dn':
            self.phis = np.random.uniform(0, 2*np.pi, size=(2**self.nqbits))
        elif self.gates_name=='MG':
            for i in range(self.num_gates):
                G = self.matchgate()
                self.gates_set.append(G)
                qbit1 = random.sample(qubit_idx,1)[0]
                qubit_idx2 = qubit_idx.copy()
                qubit_idx2.remove(qbit1)
                qbit2 = random.sample(qubit_idx2,1)[0]
                self.qubits_set.append([qbit1, qbit2])

                
    def initialization(self, initial_state):
        # 1. INITIALIZATION
        # Define initial state
        initial_state = initial_state.round(6)
        initial_state/=np.sqrt(np.sum(initial_state**2))

        # Define qiskit circuit to initialize quantum state
        self.nqbits = int(np.log2(initial_state.shape[0]))
        qc = QuantumCircuit(self.nqbits)
        qc.initialize(initial_state, list(range(self.nqbits)))
        return qc

    def apply_G_gates(self, qc):
        # Apply random gates to random qubits
        for i in range(self.num_gates):
            # Select random gate
            # Select random gate
            gate = self.gates_set[i]
            if gate=='CNOT': # For 2-qubit gates
                # Select qubit 1 and 2 (different qubits)
                qbit1, qbit2 = self.qubits_set[i]
                # Apply gate to qubits
                qc.cx(qbit1, qbit2) 
            else: # For 1-qubit gates
                # Select qubit
                qbit = self.qubits_set[i][0]
                if gate=='X':# Apply gate
                    qc.x(qbit) 
                if gate=='S':
                    qc.s(qbit) 
                if gate=='H':
                    qc.h(qbit) 
                if gate=='T':
                    qc.t(qbit) 
    
    def apply_matchgates(self, qc):
        for j in range(self.nqbits):
            qc.h(j)
        for i in range(self.num_gates):
            gate = self.gates_set[i]
            qbit1, qbit2 = self.qubits_set[i]
            qc.unitary(gate, [qbit1, qbit2], label='MG')
            
    def matchgate(self):
        A = unitary_group.rvs(2)
        B = unitary_group.rvs(2)
        detA = np.linalg.det(A)
        detB = np.linalg.det(B)
        B = B/np.sqrt(detB)*np.sqrt(detA)
        G = np.array([[A[0,0],0,0,A[0,1]],[0,B[0,0], B[0,1],0],
                      [0,B[1,0],B[1,1],0],[A[1,0],0,0,A[1,1]]])
        return G
    
    def apply_Dn(self, qc):
        for j in range(self.nqbits):
            qc.h(j)
        # Apply Dn gate
        diagonals = np.exp(1j*self.phis)
        qc.compose(Diagonal(diagonals), inplace=True)
        
    def apply_D2(self, qc):
        for j in range(self.nqbits):
            qc.h(j)
        i=0
        for pair in self.qubits_set:
            # Apply D2 gate
            diagonals = np.diag(np.exp(1j*self.phis[i]))
            D2 = UnitaryGate(diagonals)
            qc.append(D2, [pair[0], pair[1]])
            i+=1
            
    def apply_D3(self, qc):
        for j in range(self.nqbits):
            qc.h(j)
        i=0
        for pair in self.qubits_set:
            # Apply D3 gate
            diagonals = np.diag(np.exp(1j*self.phis[i]))
            D3 = UnitaryGate(diagonals)
            qc.append(D3, [pair[0], pair[1], pair[2]])
            i+=1


    def get_observables(self):
        observables = []
        name_gate=''
        for i in range(self.nqbits):
            name_gate+= 'I' 
        for i in range(self.nqbits):
            # X
            op_nameX = name_gate[:i] + 'X' + name_gate[(i+1):]
            obs = PauliOp(Pauli(op_nameX))
            observables.append(obs)
            # Y
            op_nameY = name_gate[:i] + 'Y' + name_gate[(i+1):]
            obs = PauliOp(Pauli(op_nameY))
            observables.append(obs)
            # Z
            op_nameZ = name_gate[:i] + 'Z' + name_gate[(i+1):]
            obs = PauliOp(Pauli(op_nameZ))
            observables.append(obs)
        return observables

    def run_circuit(self, initial_state):

        # 1. INITIALIZATION
        qc = self.initialization(initial_state)
        
        # 2. DEFINE RANDOM CIRCUIT
        if self.gates_name in ['G1', 'G2', 'G3']:
            self.apply_G_gates(qc)
        elif self.gates_name=='D2':
            self.apply_D2(qc)
        elif self.gates_name=='D3':
            self.apply_D3(qc)
        elif self.gates_name=='Dn':
            self.apply_Dn(qc)
        elif self.gates_name=='MG':
            self.apply_matchgates(qc)
        else:
            print('Unknown gate')

        # Obtain circuit
        #circuit = qc.to_circ()

        # 3. DEFINE OBSERVABLES
        # Define observables to measure
        if self.observables_type=='single':
            observables = self.get_observables()

        # 4. RUN CIRCUIT
        results = []
        
        backend = BasicAer.get_backend('statevector_simulator')
        job = backend.run(transpile(qc, backend))
        qc_state = job.result().get_statevector(qc)

        for obs in observables:
            obs_mat = obs.to_spmatrix()
            expect = np.inner(np.conjugate(qc_state), obs_mat.dot(qc_state)).real
            results.append(expect)

        return np.array(results)
    
    
def pauli_group_span(name_gate, nqbits=3, num_samples=400, num_gates=20):
    coefs_pauli_group = []
    Us = []
    for j in range(num_samples):
        # Define quantum circuit
        qc_tot = QuantumCircQiskit(name_gate, num_gates=num_gates,nqbits=nqbits,observables_type = 'single')
        qc = QuantumCircuit(nqbits)
        if name_gate in ['G1', 'G2', 'G3']:
            qc_tot.apply_G_gates(qc)
        elif name_gate=='D2':
            qc_tot.apply_D2(qc)
        elif name_gate=='D3':
            qc_tot.apply_D3(qc)
        elif name_gate=='Dn':
            qc_tot.apply_Dn(qc)
        elif name_gate=='MG':
            qc_tot.apply_matchgates(qc)
            
        # Get unitary
        backend = BasicAer.get_backend('unitary_simulator')
        job = backend.run(transpile(qc, backend))
        U = job.result().get_unitary(qc)
        Us.append(U)
        
        # Get pauli matrices
        #pauli_group = pauli_basis(nqbits)
        #pauli_group_matrices = np.asarray(pauli_group.to_matrix())
        # Get coefficients in pauli_group
        #size_mat = pauli_group_matrices.shape[1]
        #coefs, _,_,_ = np.linalg.lstsq((pauli_group_matrices.reshape(-1,size_mat*size_mat)).T, U.reshape(size_mat*size_mat))
        #U_hat = np.dot(coefs,pauli_group_matrices.reshape(-1,size_mat*size_mat)).reshape(size_mat,size_mat)
        #if not np.isclose(U,U_hat).all():
        #    print('Unitaries are not the same')
        #coefs_pauli_group.append(coefs)

    return Us, 0#np.asarray(coefs_pauli_group)