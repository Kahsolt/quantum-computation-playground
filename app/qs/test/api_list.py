#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/14 

import qiskit
import qiskit.circuit.library.standard_gates.equivalence_library

Module = type(qiskit)
mods_checked = set()
name_ignored = [
  'os', 'math', 'itertools', 'inspect', 'collections', 'time', 'sys', 'pkgutil',
  'namedtuple', 'fractions', 'logging', 'pickle', 'pathlib', 'random', 'mp', 'string',
  'numpy', 'np', 'pygments', 'multiprocessing', 'lazy_tester',
  'namespace',
  'pulse', 'providers', 'rx',
  'qasm', 'qobj', 'transpiler', 'dagcircuit', 'compiler',
]
import typing   ; name_ignored.extend([name for name in dir(typing)])
import pyqpanda ; name_ignored.extend([name for name in dir(pyqpanda)])
import abc      ; name_ignored.extend([name for name in dir(abc)])
name_ignored.remove('utils')


def walk(mod:Module, path:str, fh):
  mods_checked.add(mod)
  for name in dir(mod):
    if name.startswith('_'): continue
    if name in name_ignored: continue
    try: obj = getattr(mod, name)
    except: continue
    subpath = f'{path}.{name}'
    if isinstance(obj, Module):
      if obj not in mods_checked:
        walk(obj, subpath, fh)
    else:
      fh.write(subpath + '\n')

with open('api_list.txt', 'w', encoding='utf-8') as fh:
  walk(qiskit, 'qiskit', fh)
