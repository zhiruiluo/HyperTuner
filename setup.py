import sys

import setuptools

import versioneer

with open("README.md") as fh:
    long_description = fh.read()
packages = setuptools.find_namespace_packages(include=["hyper_tuner*"])
print("PACKAGES FOUND:", packages)
print(sys.version_info)

with open("requirements.txt") as req_file:
    install_requires = req_file.read().splitlines(keepends=False)


setuptools.setup(
    name="hyper_tuner",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Zhirui Luo",
    author_email="zhirluo@gmail.com",
    description="A workflow pipeline for hyperparameter tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhiruiluo/HyperTuner",
    packages=packages,
    package_data={"hyper_tuner": ["py.typed"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=install_requires,
    setup_requires=["pre-commit"],
)