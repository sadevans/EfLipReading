from setuptools import find_namespace_packages, setup, find_packages


with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [req.rstrip("\n") for req in fh]

setup(
    name="modeling",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    # packages=["backend", "pipelines", "model"],
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
    package_data= {
        # all .dat files at any package depth
        "scripts.configs": ['**/*.yaml', '*.yaml', 'data/dataset/*.yaml', '**/**/*.yaml'],
        "scripts.model.labels": ['**/*.yaml', '*.yaml', '**/**/*.yaml', '**/**/*.yaml'],
        "scripts.data.detectors.mediapipe": ['**/*.yaml', '**/*.npy', 'detector/*.npy', '*.npy',],
        # into the data folder (being into a module) but w/o the init file
    },
    python_requires=">=3.8",
    install_requires=requirements,
)