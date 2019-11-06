from setuptools import setup, find_packages

with open("README.md") as f:
    LONG_DESC = f.read()

setup(
    name="normflow",
    version="0.1",
    description="Normalising flow model on the lattice",
    author="Michael Wilson",
    author_email="michael.wilson@ed.ac.uk",
    url="https://github.com/wilsonmr/normflow_lattice",
    long_description=LONG_DESC,
    entry_points={"console_scripts": ["normflowphifour = normflow.phi_four:main"]},
    packages=find_packages(),
    zip_safe=False,
)
