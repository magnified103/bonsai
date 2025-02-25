from setuptools import setup, find_packages

setup(
    name='bonsai-network-generator',
    version='0.1.0',
    #package_dir={"": "bonsainet"},
    #packages=find_packages(where="bonsainet"),
    packages=find_packages(include=['bonsai', 'bonsai.*'])
)