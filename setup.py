from setuptools import find_packages, setup


setup(
    name="ald-font",
    version="0.1.0",
    description="Official implementation of ALD-Font for few-shot Chinese font generation",
    packages=find_packages(),
    python_requires=">=3.8,<3.11",
    install_requires=["torch", "numpy<2", "tqdm"],
)
