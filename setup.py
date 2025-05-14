from setuptools import setup, find_packages

setup(
    name="pss",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "streamlit>=1.24.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "mplsoccer>=1.1.0",
        "msgpack>=1.0.5",
        "Pillow>=10.0.0",
        "requests>=2.31.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0",
    ],
) 