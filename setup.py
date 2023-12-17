from pathlib import Path
from setuptools import find_packages, setup


setup(
    name='gaussian-rasterizer',
    version='0.1',
    packages=find_packages(),
    install_requires = [
        'tqdm',
        'tensordict'
    ],
)
