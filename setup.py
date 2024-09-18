from setuptools import setup, find_packages

setup(
    name="text_preprocessing_assignment",
    version="0.1",
    description="Text preprocessing program for NLP assignment",
    author="Shilpa Musale",
    packages=find_packages(),
    install_requires=[
        "nltk>=3.6.2",
        "numpy>=1.21.0",
        "pandas>=1.3.1",
        "matplotlib>=3.4.2",
        ],
    # Optional: include more metadata or configurations if needed
    python_requires='>=3.6',  # Specify the minimum Python version required
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
