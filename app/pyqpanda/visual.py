#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/20 

import numpy as np
import pyqpanda as pq
import pyqpanda.Visualization as pqvis

from pyqpanda import *


qvm = CPUQVM()
qvm.init_qvm()
qv = qvm.qAlloc_many(2)
cv = qvm.cAlloc_many(1)

''' pqvis.circuit_draw '''
pqvis.draw_qprog
pqvis.circuit_draw.draw_qprog_text
pqvis.circuit_draw.draw_qprog_latex

prog = QProg() << H(qv) << CNOT(qv[0], qv[1]) << Measure(qv[0], cv[0])
print(pqvis.draw_qprog(prog, output='text'))
pqvis.draw_qprog(prog, output='pic', scale=0.8, filename='qc.png', with_logo=True)
print(draw_qprog_text(prog))
print(draw_qprog_text_with_clock(prog))
print(draw_qprog_latex(prog))
#print(draw_qprog_latex_with_clock(prog))

''' pqvis.draw_probability_map '''
pqvis.draw_probability
pqvis.draw_probability_dict

probs = {
  '00': 0.0,
  '01': 0.5,
  '10': 0.3,
  '11': 0.2,
}
pqvis.draw_probability(probs)
pqvis.draw_probability_dict(probs)   # this will ignore all zero prob items

pqvis.quantum_state_plot
pqvis.plot_state_city
pqvis.plot_density_matrix
pqvis.state_to_density_matrix

q = np.array([0.2241+0.483j , 0.8365-0.1294j])
#pqvis.plot_state_city(q, title='real/imag of rho')           # <- AttributeError
#pqvis.plot_density_matrix(pqvis.state_to_density_matrix(q))  # <- AttributeError

pqvis.bloch_plot
pqvis.plot_bloch_circuit
pqvis.plot_bloch_vector
pqvis.plot_bloch_multivector

pqvis.plot_bloch_circuit(H(qv[0]))
pqvis.plot_bloch_circuit(X(qv[0]))
pqvis.plot_bloch_circuit(Y(qv[0]))
pqvis.plot_bloch_circuit(Z(qv[0]))
pqvis.plot_bloch_circuit(RX(qv[0], 2.4))
pqvis.plot_bloch_circuit(RY(qv[0], 2.4))
pqvis.plot_bloch_circuit(RZ(qv[0], 2.4))

pqvis.plot_bloch_multivector(q)

print(pqvis.pi_check.pi_check(3.14159/2))
print(pqvis.pi_check.pi_check(3.141592/2))

# ↓↓↓ inner API, maybe no use ↓↓↓
pq.Visualization.pi_check         # float number round to pi-string
pq.Visualization.exceptions
pq.Visualization.parameterexpression
pq.Visualization.bloch            # draw-related inner module
pq.Visualization.circuit_style
pq.Visualization.matplotlib_draw
pq.Visualization.utils
