from setuptools import setup

setup(
    name="tweeteval",
    version="0.1",
    description="TweetEval benchmark. Seven heterogenous tasks framed as multi-class tweet classification.",
    url="https://github.com/graphext/tweeteval",
    author="CardiffNLP",
    author_email=None,
    license=None,
    packages=["tweeteval"],
    install_requires=[
        "scikit-learn",
    ],
    scripts=["scripts/tweeteval.py"],
    include_package_data=True,
    zip_safe=False,
)
