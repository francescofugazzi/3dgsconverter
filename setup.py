from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Read the contents of the requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='gsconverter',
    version='0.2',
    author='Francesco Fugazzi',
    #author_email='your.email@example.com',
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
        ],
    },  
)
