#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/27 

from pprint import pprint as pp 
from pyqpanda import *


def makeGHZ(qv) -> QProg:
  return QProg() << H(qv[0]) << CNOT(qv[0], qv[1]) << CNOT(qv[1], qv[2])


def runFullAmpQVM():
  # 全振幅虚拟机: 测量所有振幅
  QVM_CLASSES = [
    CPUQVM,
    CPUSingleThreadQVM,
    GPUQVM,
  ]
  for qvm_cls in QVM_CLASSES:
    print(f'>> running {qvm_cls.__name__}')

    qvm = qvm_cls()
    qvm.init_qvm()
    qv = qvm.qAlloc_many(3)
    cv = qvm.cAlloc_many(3)   # classic-condition bit
    prog = makeGHZ(qv)

    # 概率测量 (理论概率)
    pmeasure_result = qvm.prob_run_dict(prog, qv)
    pp(pmeasure_result)   # {'000': 0.5, '001': 0.0, ...}
    pmeasure_result = qvm.prob_run_list(prog, qv)
    pp(pmeasure_result)   # [0.5, 0.0, ...]
    pmeasure_result = qvm.prob_run_tuple_list(prog, qv)
    pp(pmeasure_result)   # [(0, 0.5), (1, 0.0), ...]
    
    # 蒙卡测量 (实际频率)
    prog << measure_all(qv, cv)
    measure_result = qvm.run_with_configuration(prog, cv, 10000)
    pp(measure_result)
    
    qvm.finalize()
    print()


def runPartialAmpQVM():
  # 部分振幅虚拟机: 测量部分振幅
  print(f'>> running PartialAmpQVM')

  qvm = PartialAmpQVM()
  qvm.init_qvm()
  qv = qvm.qAlloc_many(10)
  prog = makeGHZ(qv)

  # 概率测量 (理论概率)
  qvm.run(prog)
  state_index = ['0', '1', '2']
  result = qvm.pmeasure_subset(state_index)   # => 得到复数振幅
  print(result)
  
  qvm.finalize()
  print()


def runSingleAmpQVM():
  # 单振幅虚拟机: 测量单个振幅
  print(f'>> running SingleAmpQVM')

  qvm = SingleAmpQVM()
  qvm.init_qvm()
  qv = qvm.qAlloc_many(10)
  prog = makeGHZ(qv)

  # 概率测量 (理论概率)
  qvm.run(prog, qv)
  result = qvm.pmeasure_dec_index('0')   # => 得到复数振幅，只能测一次
  print(result)
  
  qvm.finalize()
  print()


def runFullAmpQVMNoise():
  # 噪声虚拟机 (暂只支持CPUQVM)
  print(f'>> running runFullAmpQVMNoise')

  qvm = NoiseQVM()
  qvm.init_qvm()
  qv = qvm.qAlloc_many(10)
  cv = qvm.cAlloc_many(10)
  prog = makeGHZ(qv)
  prog << measure_all(qv, cv)

  qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.HADAMARD_GATE, 0.5)
  
  # 蒙卡测量 (实际频率)
  result = qvm.run_with_configuration(prog, cv, 10000)
  print(result)
  
  qvm.finalize()
  print()


def runRand16():
  qvm = CPUQVM()
  qvm.init_qvm()
  qubits = qvm.qAlloc_many(4)
  cbits = qvm.cAlloc_many(4)
  prog = QProg()
  prog << H(qubits) << measure_all(qubits, cbits)
  result = qvm.run_with_configuration(prog, cbits, 10000)
  pp(result)

  qvm.finalize()
  print()


if __name__ == '__main__':
  runFullAmpQVM()
  runPartialAmpQVM()
  runSingleAmpQVM()
  runFullAmpQVMNoise()

  runRand16()
