#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import setuptools

DESCR = """Multitask Random Forest Variable Importance.""".strip()
LONG_DESCR = """The Multitask Variable Importance (Multitask VI) is a modified version of the permuted variable importance score for Random Forests. Essentially, for a Random Forest trained simultaneously for multiple response vectors, it allows the inference of variable importance scores per variable and per task. For more information, see our manuscript (Aben et al, submitted, https://doi.org/10.1101/243568), where we applied this score to a dataset where each tasks corresponds to a drug combination (and hence the Multitask VI is called the Drug combination specific Variable Importance, or DVI, there).""".strip()

INSTALL_REQUIRES = ['pandas>=0.18.1', 'scikit-learn>=0.18', 'tqdm']
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
