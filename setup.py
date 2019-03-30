import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AMSIMP",
    version="0.0.1",
    author="Conor Casey",
    author_email="conorcaseyc@icloud.com",
    description="Model for Atmospheric Motion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://amsimp.github.io",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['scipy', 'astropy', 'matplotlib','numpy']
)