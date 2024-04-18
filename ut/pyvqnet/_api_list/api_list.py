#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/14 

import pyvqnet

Module = type(pyvqnet)
mods_checked = set()
name_ignored = [
  'os', 'math', 'itertools', 'inspect', 'collections', 'time',
  'namedtuple', 'fractions', 'logging', 'pickle', 'pathlib', 'random',
  'numpy', 'np',
  'pyqpanda', 'pq', 'Visualization', 'Operator',
  'matplotlib', 'mpl', 'plt', 
  'bloch', 'bloch_plot', 'proj3d', 'patches', 
  'circuit_draw', 'draw_probability_map', 'matplotlib_draw', 'quantum_state_plot',
  '_core', '_vqnet', 'pywrap', 'vqnet_core',
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
  walk(pyvqnet, 'pyvqnet', fh)
