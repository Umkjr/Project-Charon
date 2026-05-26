from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Define the compiled C++ extension module
ext_modules = [
    Pybind11Extension(
        "charon_core",
        sources=["predictive_core.cpp", "charon_pybind.cpp", "charon_io.cpp"],
        cxx_std=17,
    ),
]

setup(
    name="charon_core",
    version="1.0.0",
    author="Charon Systems Architect",
    description="Python bindings for Project Charon Predictive Memory Core",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
