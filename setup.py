import os 
from setuptools import setup, find_packages

#Use README file for long description of the project

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def parse_requiremetns(fname):
    with open(fname) as f:
        required = f.read().splitlines()
    return required
    
setup(
    name = "DDoS Detetcion with Machine Learning",
    author = "Akhil Singh Rana",
    author_email = "er.akhil.singh.rana@gmail.com",
    description = ("A machine learning approach to DDoS attack detection in Networking"),
    long_description = read("README.md"),
    install_requires = parse_requiremetns("requirements.txt"),    
    packages=find_packages(),
)