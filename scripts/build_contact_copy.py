#!/usr/bin/env python3
"""Build the small read-only contact copier for the installed Isaac Python.

Run with Isaac's Python. Build-only dependency: pybind11==2.13.6 in
.deps/pybind11-2.13.6, isolated from the shared Isaac environment.
"""
from pathlib import Path
import subprocess
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
ISAAC = Path(sysconfig.get_paths()['purelib']) / 'isaacsim'
HEADERS = ROOT / '.deps/pybind11-2.13.6/pybind11/include'
MODULE = ROOT / 'src/kcg_connector/isaac/carts_v2'
SOURCE = MODULE / 'native/contact_copy.cpp'
TARGET = MODULE / ('_contact_copy_native' + sysconfig.get_config_var('EXT_SUFFIX'))
binding = next(ISAAC.glob('extscache/omni.physx-110.1.13*/omni/physx/bindings/_physx*.so'))
abi = b'__pybind11_internals_v5_gcc_libstdcpp_cxxabi1016__'
if abi not in binding.read_bytes():
    raise RuntimeError('This native copier requires the checked PhysX pybind ABI')
if not (HEADERS / 'pybind11/pybind11.h').is_file():
    raise RuntimeError('Install the declared local pybind11 2.13.6 build dependency')
command = ['g++', '-O3', '-shared', '-std=c++17', '-fPIC', '-fvisibility=hidden',
           '-DPYBIND11_BUILD_ABI="_cxxabi1016"',
           '-I'+str(HEADERS), '-I'+sysconfig.get_paths()['include'],
           '-I'+str(ISAAC/'kit/dev/include'), str(SOURCE), '-o', str(TARGET)]
subprocess.run(command, check=True)
print(TARGET)
