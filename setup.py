from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

compilation_args = ['-fPIC', '-std=c++20', '-fopenmp']  # Добавлен флаг -fPIC
link_args = ['-fopenmp']

ext_modules = [
   Extension(
    "spin_chain",
    ["spin_chain.pyx"],
    extra_compile_args=['-fPIC', '-std=c++20', '-fopenmp'],
    extra_link_args=['-fopenmp'],
    language='c++',
    extra_objects=["build/liblibrary.a"],  
    include_dirs=[numpy.get_include()],
    libraries=['library'], 
    library_dirs=['build'],
    runtime_library_dirs=['build'],
    )
]

setup(
    ext_modules=cythonize(ext_modules, annotate=True)
)
