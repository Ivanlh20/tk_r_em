from os import path
from setuptools import setup, find_packages
import pkg_resources

_dir = path.abspath(path.dirname(__file__))

with open(path.join(_dir, 'tk_r_em', 'version.py')) as f:
    exec(f.read())

# Check for the installed version of TensorFlow
try:
    tf_version = pkg_resources.get_distribution("tensorflow").version
except pkg_resources.DistributionNotFound:
    tf_version = None

# Define the base requirements
install_requires = [
    'h5py',
]

# Adjust the requirements based on TensorFlow version
if tf_version and tf_version.startswith('2.10'):
    install_requires.extend([
        'numpy>=1.20.0,<1.24.0',
        'matplotlib>=3.1.0,<3.5.0'
    ])
else:
    install_requires.extend([
        'numpy',
        'matplotlib'
    ])

setup(
    name=__name__,
    version=__version__,
    description=__description__,
    url=__url__,
    author=__author__,
    author_email=__author_email__,
    license=__license__,
    packages=find_packages(),

    project_urls={
        'Repository': __url__,
    },

    install_requires=install_requires,
    
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    
    include_package_data=True
)