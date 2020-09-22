from setuptools import setup, find_packages
import glob

scripts = glob.glob("./stilutils/bin/*")
with open("requirements.txt", "r") as fin:
    fin = fin.read()
    requirements = fin.replace("==", ">=").split("\n")

setup(
    name="stilutils",
    description="Module full of CLI tools to support your SSX/SFX data processing",
    author="Egor Marin",
    author_email="marin@phystech.edu",
    # external packages as dependencies
    install_requires=requirements,
    scripts=scripts,
    packages=find_packages(),
)
