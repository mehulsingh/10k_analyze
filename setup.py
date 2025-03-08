from setuptools import setup, find_packages

setup(
    name="tenk_toolkit",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A toolkit for analyzing SEC 10-K filings",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/10k-analysis-toolkit",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "plotly>=5.3.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.6.0",
        "spacy>=3.2.0",
        "wordcloud>=1.8.0",
        "textblob>=0.15.3",
        "beautifulsoup4>=4.10.0",
        "requests>=2.26.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "sphinx>=4.2.0",
            "black>=21.9b0",
            "flake8>=4.0.0",
        ],
    },
)