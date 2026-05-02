from pathlib import Path
import re

from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Read the contents of the requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()


def read_version():
    version_file = Path(__file__).with_name('gsconverter').joinpath('version.py')
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", version_file.read_text(encoding='utf-8'))
    if not match:
        raise RuntimeError("Unable to find package version")
    return match.group(1)

setup(
    name='gsconverter',
    version=read_version(),
    author='Francesco Fugazzi',

    description='3D Gaussian Splatting Converter',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/francescofugazzi/3dgsconverter',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            '3dgsconverter=gsconverter.main:main',
            'gsconverter=gsconverter.main:main',
            '3dgsconv=gsconverter.main:main',
            'gsconv=gsconverter.main:main',
        ],
    },  
)
