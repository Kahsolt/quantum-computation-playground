#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/14 

import sys
import pyqpanda
from distutils import sysconfig

site_packages_path = sysconfig.get_python_lib().lower()
print('site-package path:', site_packages_path)

Module = type(pyqpanda)
mods_checked = set()
name_ignored = [
  '_core', '_vqnet', 'pywrap', 'sm4',
  'numpy', 'np', 'sparse', 
  'matplotlib', 'mpl', 'plt', 'mcolors', 'animation', 'patches',
  'requests', 
]
import typing ; name_ignored.extend([name for name in dir(typing)])
import abc    ; name_ignored.extend([name for name in dir(abc)])


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
      if not obj.__file__.lower().startswith(site_packages_path): continue
      if obj in mods_checked: continue
      walk(obj, subpath, fh)
    else:
      fh.write(subpath + '\n')

with open('api_list.txt', 'w', encoding='utf-8') as fh:
  walk(pyqpanda, 'pyqpanda', fh)

print('Done')
