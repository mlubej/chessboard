import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genetic_algorithm",
    version="0.0.1",
    author="Matic Lubej",
    author_email="lubej.matic@gmail.com",
    description="TODO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlubej/genetic_chessboard",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

