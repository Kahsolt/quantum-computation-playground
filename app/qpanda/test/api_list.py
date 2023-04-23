#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/14 

import pyqpanda

Module = type(pyqpanda)
mods_checked = set()
name_ignored = [
  'os', 'math', 'itertools', 'inspect', 'collections', 'time',
  'namedtuple', 'fractions', 'logging', 'pickle', 'pathlib', 'random',
  'numpy', 'np',
  'matplotlib', 'mpl', 'plt', 
  '_core', '_vqnet', 'pywrap',
]
import typing   ; name_ignored.extend([name for name in dir(typing)])
import abc      ; name_ignored.extend([name for name in dir(abc)])


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
  walk(pyqpanda, 'pyqpanda', fh)
