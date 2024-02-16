#! /usr/bin/env python3

import os

try:
    from setuptools import find_packages, setup
except AttributeError:
    from setuptools import find_packages, setup

NAME = 'shadow-hybrid-methods'

VERSION = '1.0.4'
ISRELEASED = True

DESCRIPTION = 'Hybrid Methods, combining raytracing with wave optics'
README_FILE = os.path.join(os.path.dirname(__file__), 'README.md')
LONG_DESCRIPTION = open(README_FILE).read()
AUTHOR = 'Luca Rebuffi, Xianbo Shi'
AUTHOR_EMAIL = 'lrebuffi@anl.gov'
URL = 'https://github.com/oasys-kit/hybrid-methods'
DOWNLOAD_URL = 'https://github.com/oasys-kit/hybrid-methods'
MAINTAINER = 'Luca Rebuffi'
MAINTAINER_EMAIL = 'lrebuffi@anl.gov'
LICENSE = 'BSD'

KEYWORDS = (
    'x-ray'
    'synchrotron radiation',
    'wavefront propagation'
    'ray tracing',
    'surface metrology',
    'simulation',
)

CLASSIFIERS = (
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Environment :: Plugins',
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: '
    'GNU General Public License v3 or later (GPLv3+)',
    'Operating System :: POSIX',
    'Operating System :: Microsoft :: Windows',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
)

INSTALL_REQUIRES = (
    'setuptools',
    'numpy',
    'scipy',
    'srxraylib>=1.0.45',
    'syned',
    'wofry',
    'wofryimpl>=1.0.26',
)

SETUP_REQUIRES = (
    'setuptools',
)

PACKAGES = [
    "hybrid_methods",
    "hybrid_methods.coherence",
    "hybrid_methods.undulator",
]

PACKAGE_DATA = {
}


def setup_package():
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        url=URL,
        download_url=DOWNLOAD_URL,
        license=LICENSE,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        # extra setuptools args
        zip_safe=True,  # the package can run out of an .egg file
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        setup_requires=SETUP_REQUIRES,
    )

if __name__ == '__main__':
    setup_package()
