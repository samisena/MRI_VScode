from setuptools import setup, find_packages


setup(
    name="mri",                             #* This is how to import this package
    packages=find_packages(where="src"),    #* Automatically finds all packages in src/
    package_dir={"": "src"},                #* Tells Python "the packages are in src/"
)