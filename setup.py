#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import setuptools

DESCR = """Description.""".strip()
LONG_DESCR = """Long description.""".strip()

INSTALL_REQUIRES = ['pandas>=0.18.1', 'scikit-learn>=0.18']
EXTRAS_REQUIRE = {}

setuptools.setup(
    name='multitask_vi',
    version='0.1',
    description=DESCR,
    long_description=LONG_DESCR,
    url=
    'https://bitbucket.org/naben/drug-combination-specific-variable-importance',
    author='Julian de Ruiter',
    author_email='julianderuiter@gmail.com',
    license='MIT license',
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'mt-train = multitask_vi.main.train:main',
            'mt-vi = multitask_vi.main.multitask_vi:main'
        ]
    },
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    zip_safe=False,
    classifiers=[])
