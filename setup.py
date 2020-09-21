from setuptools import setup
import glob

scripts = glob.glob("./still-utils/bin/*er") + glob.glob("./still-utils/bin/*.py")
with open("requirements.txt", "r") as fin:
    requirements = fin.split("\n")

setup(
    name="still-utils",
    version="0.1",
    description="Module full of CLI tools to support your SSX/SFX data processing",
    author="Egor Marin",
    author_email="marin@phystech.edu",
    # external packages as dependencies
    install_requires=requirements,
    scripts=scripts,
)
