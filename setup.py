#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="tweeteval",
    version="0.1",
    description="TweetEval benchmark. Seven heterogenous tasks framed as multi-class tweet classification.",
    url="https://github.com/graphext/tweeteval",
    author="CardiffNLP",
    author_email=None,
    license=None,
    packages=find_packages(),
    install_requires=["scikit-learn", "typer"],
    entry_points={
        "console_scripts": ["tweeteval = tweeteval.cli:CLI"],
    },
    include_package_data=True,
    zip_safe=False,
)
