from pathlib import Path
from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / "README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="duonlabs",
    version="0.0.1",
    description="Python package to interact with Duonlabs's API.",
    author="Duon labs",
    author_email="contact@duonlabs.com",
    url="https://github.com/duonlabs/duonlabs",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["duonlabs"],
    install_requires=["numpy", "requests"],
    extras_require={
        "ccxt": ["ccxt"],
        "dev": ["pytest", "ruff"]
    },
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
