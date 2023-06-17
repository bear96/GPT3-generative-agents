from setuptools import setup, find_packages

# Read the requirements from requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='dorito',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
)
