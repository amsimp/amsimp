import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="amsimp",
    version="0.0.2",
    author="Conor Casey",
    author_email="conorcaseyc@icloud.com",
    description="Model for Atmospheric Dynamics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://amsimp.github.io",
    download_url="https://github.com/amsimp/amsimp.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["scipy", "astropy", "matplotlib", "numpy", "cartopy"],
)
