from setuptools import setup

setup(
    name="airfoil",
    version="0.6.2",
    description="Just another airfoil manipulation package",
    packages=["airfoil"],
    author="Christian Hauschel",
    zip_safe=False,
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "cst",
        "polygon_math",
        "splines @ git+https://github.com/christianhauschel/splines.git",
    ],
)
