from setuptools import setup, find_packages

setup(
    name="pss",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "streamlit>=1.45.0",
        "pandas>=2.2.3",
        "matplotlib>=3.9.4",
        "mplsoccer>=1.4.0",
        "msgpack>=1.1.0",
        "numpy>=2.0.2",
        "plotly>=6.0.1",
        "seaborn>=0.13.2",
        "python-dotenv>=1.0.1",
        "pathlib>=1.0.1",
    ],
    python_requires=">=3.8",
) 