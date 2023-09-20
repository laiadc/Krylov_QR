# Quantum reservoir complexity by Krylov evolution approach  

Quantum reservoir computing algorithms recently emerged as a standout approach in the development of successful methods for the NISQ era, because of its superb performance and compatibility with current quantum devices.  By harnessing the properties and dynamics of a quantum system, quantum reservoir computing effectively uncovers hidden patterns in data.  However, the design of the quantum reservoir is crucial to this end, in order to ensure an optimal performance of the algorithm. In this work, we introduce a precise quantitative method, with a strong physical foundations based on the Krylov evolution, to assess the wanted good performance in machine learning tasks. Our results show that the Krylov approach to complexity strongly correlates with quantum reservoir performance, making of it a powerful tool in the quest of optimally designed of quantum reservoirs, which will pave the road to the implementation of successful quantum machine learning methods.

The results are illustrated in a set of [Jupyter](https://jupyter.org/) notebooks for easier comprehension. We also provide a python script to run the experiments. The whole code is the result of the work in  this paper. Any contribution or idea to continue the lines of the proposed work will be very welcome.

## Notebooks

### [krylov_Ising.ipynb](https://github.com/laiadc/Krylov_QR/blob/main/krylov_Ising.ipynb)  
The goal of this notebook is to calculate the Krylov complexity for the longitudinal-transverse field Ising model. Given a unitary evolution $U$, we construct an effective Hamiltonian $H_\text{eff} = -i/T \log U$. We then compute the Krylov statistics with this effective operator. In this notebook we validate this approach for three different time scales: the Heisemberg time, the scrambling time and a shorter time.

### [krylov_standard_map.ipynb](https://github.com/laiadc/Krylov_QR/blob/main/krylov_standard_map.ipynb) 
The goal of this notebook is to calculate the Krylov complexity in the standard map. In this case, there is no Hamiltonian operator describing the unitary evolution, but rather just a unitary operator.
We will calculate the Krylov complexity by taking the logarithm of such unitary, and we will prove that the energy level statistics, the Krylov coefficients and complexity are correctly reproduced with this approach.

### [krylov_QR.ipynb](https://github.com/laiadc/Krylov_QR/blob/main/krylov_QR.ipynb) 
The goal of this notebook is to calculate the Krylov complexity in different families of quantum reservoirs, which have different complexity according to the majorization criterion. We have observed that, in small Hilbert spaces, the distribution of the Unitaries in the Pauli space is different for each family. In this way, the families with higher complexity span the Pauli space uniformly, while those with lower complexity span only a subspace of the Pauli space. Our aim is to show that the krylov complexity is able to capture this difference in complexity.

## Contributions

Contributions are welcome!  For bug reports or requests please [submit an issue](https://github.com/laiadc/Krylov_QR/issues).

## Contact  

Feel free to contact me to discuss any issues, questions or comments.

* GitHub: [laiadc](https://github.com/laiadc)
* Email: [laia@ingenii.dev](laia@ingenii.dev)

### BibTex reference format for citation for the Code
```
@misc{KrylovDomingo,
title={Quantum reservoir complexity by Krylov evolution approach  },
url={https://github.com/laiadc/Krylov_QR},
note={GitHub repository containing a study of the Krylov complexity in quantum reservoirs.},
author={Laia Domingo},
  year={2023}
}
```
