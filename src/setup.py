from setuptools import setup, find_packages

__version__ = "_"
exec(open('flat_bug/_version.py').read())

setup(
    name='flat_bug',
    version=__version__,
    long_description=__doc__,
    packages=find_packages(),
    scripts=['bin/fb_train.py',
             'bin/fb_predict.py',
             'bin/fb_prepare_data.py',
             ],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'ultralytics',
        'numpy',
        'opencv_python',
        'torch',
        'shapely',
        'torchvision',
        'sklearn'],
    extras_require={
        'test': ['nose', 'pytest', 'pytest-cov', 'codecov', 'coverage'],
    },
    test_suite='nose.collector'
)

