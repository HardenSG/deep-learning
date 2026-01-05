from setuptools import setup, find_packages

setup(
    name="quant-dl-system",
    version="0.1.0",
    description="A股量化深度学习系统",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "akshare>=1.12.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "sqlalchemy>=2.0.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
        "scikit-learn>=1.3.0",
        "apscheduler>=3.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "jupyter>=1.0.0",
        ]
    },
)
