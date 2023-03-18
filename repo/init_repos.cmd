@ECHO OFF

REM Qiskit
git clone https://github.com/Qiskit/qiskit
git clone https://github.com/Qiskit/qiskit-machine-learning
git clone https://github.com/Qiskit/qiskit-terra

REM Cirq
git clone https://github.com/quantumlib/cirq

REM ProjectQ
git clone https://github.com/ProjectQ-Framework/ProjectQ

REM TensorFlow quantum
git clone https://github.com/tensorflow/quantum tensorflow-quantum

REM Torch quantum
git clone https://github.com/mit-han-lab/torchquantum

REM PaddlePaddle quantum (Baidu)
git clone https://github.com/PaddlePaddle/Quantum PaddlePaddle-Quantum

REM Mind quantum (Huawei)
git clone https://gitee.com/mindspore/mindquantum

REM Quantum-programming-textbook (OriginQ)
git clone https://github.com/OriginQ/Quantum-programming-textbook

REM QPanda (OriginQ)
git clone https://github.com/OriginQ/QPanda-2

REM VQNet (OriginQ) is close-source, but we found some repos using it
git clone https://github.com/RIvance/QLSTM-VQNet

REM pyChemiQ (OriginQ)
git clone https://github.com/OriginQ/pyChemiQ

REM Python-for-Tensor-Network-Tutorial
git clone https://github.com/ranshiju/Python-for-Tensor-Network-Tutorial

REM TensorNetworkClassLibary
git clone https://github.com/ranshiju/TensorNetworkClassLibary

REM Tiny-Q
git clone https://github.com/Kahsolt/Tiny-Q


ECHO Done!
ECHO.

PAUSE
