from setuptools import setup, find_packages

from tk_r_em import  __name__,__version__,__description__,__url__
from tk_r_em import  __author__, __author_email__, __credits__, __license__

setup(
    name=__name__,
    version=__version__,    
    description=__description__,
    url=__url__,
    author=__author__,
    author_email=__author_email__,
    credits=__credits__,
    license=__license__,
    packages=find_packages(),
 
    project_urls={
        'Repository': __url__,
    },
         
    install_requires=[
                    'numpy', 
                    'h5py', 
                    'matplotlib'],
    
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