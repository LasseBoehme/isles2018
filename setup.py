# coding=utf-8
# Diese Datei wird nur für die Google Cloud ML benötigt, um externe Pakete auf der VM zu installieren.

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ["nibabel>=2.3.0", "scikit-image>=0.14.0", "keras>=2.2.0"]

setup(
    name="gctest.trainer",
    version="0.1",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="My training application package."
)

