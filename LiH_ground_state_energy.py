# Use PySCF, a classical computational chemistry software
# package, to compute the one-body and two-body integrals in
# molecular-orbital basis, necessary to form the Fermionic operator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
import warnings
warnings.filterwarnings('ignore')
driver = PySCFDriver(atom='Li .0 .0 .0; H .0 .0 1.595',
                     unit=UnitsType.ANGSTROM,
                     basis='sto3g')
molecule = driver.run()

# Build the qubit operator, which is the input to the VQE algorithm in Aqua
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType
operator =  Hamiltonian(
                transformation=TransformationType.FULL,
                qubit_mapping=QubitMappingType.PARITY, # Other choices: JORDAN_WIGNER, BRAVYI_KITAEV
                two_qubit_reduction=True,
                freeze_core=True,
                orbital_reduction=[-3, -2])
qubit_op, _ = operator.run(molecule)

# Control group baseline results
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
result = NumPyMinimumEigensolver(qubit_op).run()
result = operator.process_algorithm_result(result)
print("====================================================")
print('Ground state energy (classical)  : {:.12f}'.format(result.energy))
print(result)
print("====================================================")

# setup a classical optimizer for VQE
from qiskit.aqua.components.optimizers import SPSA
optimizer = SPSA()

# setup the variational form for VQE
from qiskit.circuit.library import TwoLocal
var_form = TwoLocal(qubit_op.num_qubits, ['ry', 'rz'], 'cz', reps=5, entanglement='full')
print("Ansatz quantum circuit:")
print(var_form)

# setup and run VQE
from qiskit.aqua.algorithms import VQE
algorithm = VQE(qubit_op, var_form, optimizer)

# set the backend for the quantum computation
from qiskit import Aer
backend = Aer.get_backend('statevector_simulator')

result = algorithm.run(backend)
result = operator.process_algorithm_result(result)
print("====================================================")
print('Ground state energy (quantum)  : {:.12f}'.format(result.energy))
print(result)
print("====================================================")
