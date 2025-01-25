from setuptools import setup, find_packages


setup(
    name="mri",  # This is how other developers will import your package
    packages=find_packages(where="src"),  # Automatically finds all packages in src/
    package_dir={"": "src"},  # Tells Python "the packages are in src/"
)