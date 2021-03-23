#!/usr/bin/env python3
# -*-encoding: utf-8-*-

# created: 23.03.2021
# Excusa. Quod scripsi, scripsi.

# by d.zashkonyi

from distutils.core import setup
from Cython.Build import cythonize

setup (
    name = 'autogrp_cython',
    ext_modules = cythonize(["_autogrp_cython/*.pyx"], compiler_directives={'language_level' : "3"}),

)