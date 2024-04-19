#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/14 

import warnings ; warnings.filterwarnings(action='ignore', category=DeprecationWarning)

import sys
import pyvqnet
from distutils import sysconfig

site_packages_path = sysconfig.get_python_lib().lower()
print('site-package path:', site_packages_path)

Module = type(pyvqnet)
mods_checked = set()
name_ignored = [
  '_core', '_vqnet', 'pywrap', 'vqnet_core',
  'pyqpanda', 'pq', 'Visualization', 'Operator',
  'bloch', 'bloch_plot', 'proj3d', 'circuit_draw', 'draw_probability_map', 'matplotlib_draw', 'quantum_state_plot',
  'numpy', 'np', 'scipy', 'sparse', 
  'matplotlib', 'mpl', 'plt', 'mcolors', 'animation', 'patches',
  'bson',
]
import typing   ; name_ignored.extend([name for name in dir(typing)])
import pyqpanda ; name_ignored.extend([name for name in dir(pyqpanda)])
import abc      ; name_ignored.extend([name for name in dir(abc)])
name_ignored.remove('utils')


def walk(mod:Module, path:str, fh):
  mods_checked.add(mod)
  for name in dir(mod):
    if name.startswith('_'): continue
    if name in sys.builtin_module_names: continue
    if name in name_ignored: continue
    try: obj = getattr(mod, name)
    except: continue
    subpath = f'{path}.{name}'
    if isinstance(obj, Module):
      if (hasattr(obj, '__file__') and 
          obj.__file__ and 
          not obj.__file__.lower().startswith(site_packages_path)): continue
      if obj in mods_checked: continue
      walk(obj, subpath, fh)
    else:
      fh.write(subpath + '\n')

with open('api_list.txt', 'w', encoding='utf-8') as fh:
  walk(pyvqnet, 'pyvqnet', fh)

print('Done')
