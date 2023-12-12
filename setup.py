import setuptools

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')


setuptools.setup(name="netflow",
                 version="0.0.ddv",
                 author="Rena Elkin"
                 description="Toolbox to compute information flow on a network and correlations between network modules",
                 install_requires=install_requires,
                 packages=setuptools.find_packages(),
)    
