#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/20

# API list of pyqpanda v3.8.3.3
# https://pyqpanda-toturial.readthedocs.io/zh/latest/index.html

# classical
from pyqpanda import CBit, ClassicalCondition, OriginCMem
from pyqpanda import add, sub, div, mul, assign, equal

# qubit
from pyqpanda import Qubit, QVec, PhysicalQubit, OriginQubitPool

# gate
from pyqpanda import QGate, I, H, X, Y, Z, RX, RY, RZ, P, S, T, U1, U2, U3, U4, X1, Y1, Z1, RXX, RYY, RZZ, RZX, CNOT, CP, CR, CU, CZ, SWAP, iSWAP, SqiSWAP, Toffoli, GateType
from pyqpanda import QDouble, QOracle, MS, BARRIER
from pyqpanda import matrix_decompose, matrix_decompose_paulis, ldd_decompose, DecompositionMode, decompose_multiple_control_qgate, transform_to_base_qgate, transfrom_pauli_operator_to_matrix, virtual_z_transform
from pyqpanda import QReset, Reset, QMeasure, Measure, measure_all, pmeasure, PMeasure, pmeasure_no_index, PMeasure_no_index

# circuit & prog
from pyqpanda import QCircuit, create_empty_circuit, CreateEmptyCircuit
from pyqpanda import QProg, QIfProg, QWhileProg, create_empty_qprog, CreateEmptyQProg, create_if_prog, CreateIfProg, create_while_prog, CreateWhileProg, ClassicalProg, NodeIter, NodeInfo, NodeType
from pyqpanda import Fusion, QCircuitOPtimizerMode, SingleGateTransferType, DoubleGateTransferType
from pyqpanda import get_clock_cycle, get_qprog_clock_cycle, get_qgate_num, count_gate, count_qgate_num, count_prog_info
from pyqpanda import cast_qprog_qcircuit, cast_qprog_qgate, cast_qprog_qmeasure

# visualize
from pyqpanda import draw_qprog_text, draw_qprog_text_with_clock, draw_qprog_latex, draw_qprog_latex_with_clock, LatexMatrix, LATEX_GATE_TYPE
from pyqpanda.Visualization.quantum_state_plot import plot_state_city, plot_density_matrix, state_to_density_matrix
from pyqpanda.Visualization.bloch_plot import plot_bloch_circuit, plot_bloch_vector, plot_bloch_multivector
from pyqpanda.Visualization.circuit_draw import draw_qprog, draw_circuit_pic, show_prog_info_count
from pyqpanda.Visualization.draw_probability_map import draw_probability, draw_probability_dict

# simulator
from pyqpanda import QuantumMachine, CPUQVM, CPUSingleThreadQVM, GPUQVM, SingleAmpQVM, PartialAmpQVM, MPSQVM, DensityMatrixSimulator, SparseQVM, NoiseQVM, NoiseModel, Noise, Stabilizer, QMachineType, BackendType
from pyqpanda import init, finalize, init_quantum_machine, destroy_quantum_machine, getstat, get_qstate
from pyqpanda import qAlloc, qAlloc_many, qFree, qFree_all, cAlloc, cAlloc_many, cFree, cFree_all
from pyqpanda import getAllocateCMem, get_allocate_cmem_num, getAllocateQubitNum, get_allocate_cbits, get_allocate_qubit_num, get_allocate_qubits, get_all_used_qubits, get_all_used_qubits_to_int
from pyqpanda import prob_run_dict, prob_run_tuple_list, prob_run_list, run_with_configuration, get_prob_dict, get_prob_list, get_tuple_list, directly_run, quick_measure

# real-chip
from pyqpanda import QCloudService, QCloud, QCloudTaskConfig, ChipID, real_chip_type, QPilotOSService, QPilotMachine, PilotNoiseParams, ErrorCode
from pyqpanda import sabre_mapping, OBMT_mapping, topology_match, is_match_topology, quantum_chip_adapter
from pyqpanda import calculate_quantum_volume, single_qubit_rb, double_qubit_rb, double_gate_xeb

# variational & optim
from pyqpanda import var, expression
from pyqpanda import sum, dot, poly, sin, cos, tan, asin, acos, atan, exp, log, inverse, sigmoid, softmax, crossEntropy, dropout, transpose, stack, eval
if 'variational stuff':
  from pyqpanda import (
    VariationalQuantumCircuit,
    VariationalQuantumGate,
    VariationalQuantumGate_I,
    VariationalQuantumGate_H,
    VariationalQuantumGate_X,
    VariationalQuantumGate_Y,
    VariationalQuantumGate_Z,
    VariationalQuantumGate_RX,
    VariationalQuantumGate_RY,
    VariationalQuantumGate_RZ,
    VariationalQuantumGate_S,
    VariationalQuantumGate_T,
    VariationalQuantumGate_U1,
    VariationalQuantumGate_U2,
    VariationalQuantumGate_U3,
    VariationalQuantumGate_U4,
    VariationalQuantumGate_X1,
    VariationalQuantumGate_Y1,
    VariationalQuantumGate_Z1,
    VariationalQuantumGate_CRX,
    VariationalQuantumGate_CRY,
    VariationalQuantumGate_CRZ,
    VariationalQuantumGate_CNOT,
    VariationalQuantumGate_CR,
    VariationalQuantumGate_CU,
    VariationalQuantumGate_CZ,
    VariationalQuantumGate_SWAP,
    VariationalQuantumGate_iSWAP,
    VariationalQuantumGate_SqiSWAP,
    VQG_I_batch,
    VQG_H_batch,
    VQG_X_batch,
    VQG_Y_batch,
    VQG_Z_batch,
    VQG_S_batch,
    VQG_T_batch,
    VQG_X1_batch,
    VQG_Y1_batch,
    VQG_Z1_batch,
    VQG_U1_batch,
    VQG_U2_batch,
    VQG_U3_batch,
    VQG_U4_batch,
    VQG_CNOT_batch,
    VQG_CZ_batch,
    VQG_CU_batch,
    VQG_SWAP_batch,
    VQG_iSWAP_batch,
    VQG_SqiSWAP_batch,
  )
from pyqpanda import Optimizer, VanillaGradientDescentOptimizer, MomentumOptimizer, AdaGradOptimizer, AdamOptimizer, RMSPropOptimizer, AbstractOptimizer, OptimizerFactory, OptimizerType, QOptimizationResult

# ansatz
from pyqpanda import Encode, amplitude_encode
from pyqpanda import hadamard_circuit, random_qcircuit, Ansatz, AnsatzGate, AnsatzGateType
from pyqpanda import random_qprog, fill_qprog_by_I, apply_QGate, flatten, bind_data, bind_nonnegative_data
if 'transcription stuff':
  from pyqpanda import (
    # QProg <-> OriginIR
    transform_qprog_to_originir, convert_qprog_to_originir, to_originir,
    transform_originir_to_qprog, convert_originir_to_qprog, convert_originir_str_to_qprog, originir_to_qprog,
    # QProg <-> Quil
    transform_qprog_to_quil, convert_qprog_to_quil, to_Quil,
    # QProg <-> QASAM
    convert_qprog_to_qasm,
    convert_qasm_to_qprog, convert_qasm_string_to_qprog,
    # QProg <-> Binary
    get_bin_data, get_bin_str,
    transform_qprog_to_binary, convert_qprog_to_binary, 
    transform_binary_data_to_qprog, convert_binary_data_to_qprog, bin_to_prog,
  )

# operator
from pyqpanda import QOperator, pauli_combination_replace
from pyqpanda import qop, qop_pmeasure
from pyqpanda.Operator.pyQPandaOperator import PauliOperator, FermionOperator, VarPauliOperator, VarFermionOperator, matrix_decompose_hamiltonian

# applications
from pyqpanda import QAdd, QAdder, QAdderIgnoreCarry, QSub, QMul, QMultiplier, QDiv, QDivWithAccuracy, QDivider, QDividerWithAccuracy, QComplement
from pyqpanda import isCarry, constModAdd, constModMul, constModExp, MAJ, MAJ2, UMA
from pyqpanda import QPE, QFT, Shor_factorization, iterative_amplitude_estimation
from pyqpanda import QITE, UpdateMode
from pyqpanda import Grover, Grover_search
from pyqpanda import HHLAlg, build_HHL_circuit, expand_linear_equations, HHL_solve_linear_equations
from pyqpanda import quantum_walk_alg, quantum_walk_search
from pyqpanda import em_method, QuantumStateTomography

# misc utils
from pyqpanda import get_matrix, get_unitary, get_adjacent_qgate_type
from pyqpanda import state_fidelity, average_gate_fidelity, accumulateProbability, accumulate_probabilities, accumulate_probability
from pyqpanda import deep_copy, print_matrix, expMat, QError, QResult, OriginCollection
