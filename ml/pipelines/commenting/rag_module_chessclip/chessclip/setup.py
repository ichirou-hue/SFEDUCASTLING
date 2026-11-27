""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# The repository manual is maintained in the root README.
long_description = 'ChessCLIP / OpenCLIP. See the SFEDUCASTLING root README for documentation.'

# Shared dependencies are installed from the repository root requirements.txt.
REQUIREMENTS = []
TRAINING_REQUIREMENTS = []

exec(open('src/open_clip/version.py').read())
setup(
    name='open_clip_torch',
    version=__version__,
    description='OpenCLIP',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mlfoundations/open_clip',
    author='',
    author_email='',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='CLIP pretrained',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    extras_require={
        "training": TRAINING_REQUIREMENTS,
    },
    python_requires='>=3.7',
)
