from setuptools import setup, find_packages

setup(
    name="OptunaFS",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "optuna",
    ],
    python_requires=">=3.8",
)
