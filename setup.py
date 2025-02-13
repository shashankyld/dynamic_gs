from setuptools import setup, find_packages

def read_requirements():
    """Read dependencies from requirements.txt, ignoring '-e' entries."""
    try:
        with open("requirements.txt") as f:
            return [line.strip() for line in f if not line.startswith("-e")]
    except FileNotFoundError:
        return []

setup(
    name='splat-slam',
    version='0.1',
    packages=find_packages(),
    install_requires=read_requirements(),
)

