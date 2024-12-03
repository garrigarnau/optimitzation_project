from setuptools import setup, find_packages

setup(
    name="ml_distributed",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'PyYAML>=6.0',
        'matplotlib>=3.5.0',
    ]
)