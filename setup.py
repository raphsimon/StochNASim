import pathlib

from setuptools import setup, find_packages

extras = {
    "ppo_trxl": [
        "torch>=2.0",
        "tensorboard>=2.10",
        "tyro>=0.5",
        "einops>=0.7",
        "optuna>=4.0",
        "sqlalchemy>=1.4",
        "wandb>=0.13",
        "huggingface_hub>=0.11",
    ],
    "sb3": [
        "stable-baselines3==2.4.*",
        "sb3-contrib==2.4.*",
        "rl-zoo3>=2.4",
        "optuna>=4.0",
        "tensorboard>=2.10",
    ],
    "test": [
        "pytest>=5.4",
    ],
}

extras["all"] = [item for group in extras.values() for item in group]


def get_version():
    """Gets the StochNASim version."""
    path = pathlib.Path(__file__).absolute().parent / "nasim" / "__init__.py"
    content = path.read_text()
    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


setup(
    name="stochnasim",
    version=get_version(),
    url="https://github.com/raphsimon/StochNASim",
    description=(
        "StochNASim: a stochastic, extension of NASim for benchmarking "
        "RL agents on automated network penetration testing."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Raphael Simon, Pieter Libin, Wim Mees",
    author_email="r.simon@cylab.be",
    license="MIT",
    packages=[package for package in find_packages() if package.startswith("nasim")],
    install_requires=[
        "gymnasium>=0.29,<0.30",
        "numpy<2.0.0",
        "networkx>=2.4",
        "matplotlib>=3.1",
        "pyyaml>=5.3",
        "prettytable>=0.7",
    ],
    extras_require=extras,
    python_requires=">=3.9",
    package_data={
        "nasim": ["scenarios/benchmark/*.yaml"],
    },
    project_urls={
        "Paper": "https://openreview.net/forum?id=YkUV7wfk19",
        "Source": "https://github.com/raphsimon/StochNASim",
        "Original Project (NASim)": "https://github.com/Jjschwartz/NetworkAttackSimulator",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    zip_safe=False,
)
