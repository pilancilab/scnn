from setuptools import setup, find_packages  # type: ignore

setup(
    name="scnn",
    version="0.0.1",
    author="Aaron Mishkin",
    author_email="amishkin@cs.stanford.edu",
    description="A package for fast convex optimization of two-layer neural networks.",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    package_data={"scnn": ["py.typed"]},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "numpy==1.21.3",
        "torch",
        "torchvision",
        "torchaudio",
        "cvxpy==1.1.15",
        "scikit-learn",
        "scipy",
        "tqdm",
        "opt-einsum",
        "matplotlib",
        "lab @ git+https://git@github.com/aaronpmishkin/lab@icml_2022#egg=lab",
    ],
)
