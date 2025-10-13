"""SkyPilot.

SkyPilot is a framework for easily running machine learning* workloads on any
cloud through a unified interface. No knowledge of cloud offerings is required
or expected – you simply define the workload and its resource requirements, and
SkyPilot will automatically execute it on AWS, Google Cloud Platform or
Microsoft Azure.

*: SkyPilot is primarily targeted at machine learning workloads, but it can
also support many general workloads. We're excited to hear about your use case
and would love to hear more about how we can better support your requirements -
please join us in [this
discussion](https://github.com/skypilot-org/skypilot/discussions/1016)
"""

import io
import os
import re

import setuptools

ROOT_DIR = os.path.dirname(__file__)


def find_version(*filepath):
    # Extract version information from filepath
    # Adapted from:
    #  https://github.com/ray-project/ray/blob/master/python/setup.py
    with open(os.path.join(ROOT_DIR, *filepath)) as fp:
        version_match = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]',
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError('Unable to find version string.')


def parse_footnote(readme: str) -> str:
    """Parse the footnote from the README.md file."""
    readme = readme.replace('<!-- Footnote -->', '#')
    footnote_re = re.compile(r'\[\^([0-9]+)\]')
    return footnote_re.sub(r'<sup>[\1]</sup>', readme)


install_requires = [
    'configargparse',
]

long_description = ''
readme_filepath = 'README.md'
# When sky/backends/wheel_utils.py builds wheels, it will not contain the
# README.  Skip the description for that case.
if os.path.exists(readme_filepath):
    long_description = io.open(readme_filepath, 'r', encoding='utf-8').read()
    long_description = parse_footnote(long_description)

setuptools.setup(
    # NOTE: this affects the package.whl wheel name. When changing this (if
    # ever), you must grep for '.whl' and change all corresponding wheel paths
    # (templates/*.j2 and wheel_utils.py).
    name='sky-spot',
    version=find_version('sky_spot', '__init__.py'),
    packages=setuptools.find_packages(),
    author='Zhanghao Wu',
    license='Apache 2.0',
    readme='README.md',
    description='SkyPilot: An intercloud broker for the clouds',
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=['wheel'],
    requires_python='>=3.6',
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Distributed Computing',
    ],
    project_urls={},
)
