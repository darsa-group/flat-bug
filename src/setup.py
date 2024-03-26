from setuptools import setup, find_packages

__version__ = "_"
exec(open('flat_bug/_version.py').read())

setup(
    name='flat_bug',
    version=__version__,
    long_description=__doc__,
    packages=find_packages(),
    scripts=['bin/fb_train.py',
             'bin/fb_eval.py',
             'bin/fb_predict.py',
             'bin/fb_prepare_data.py',
             'bin/fb_predict_erda.py',
             ],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'ultralytics>=8.0.225',
        'numpy',
        'pyexiftool',
        'opencv_python',
        'torch',
        'shapely',
        'torchvision',
        'scikit-learn',
        "IPython",
        "ipywidgets",
        "tqdm"],
    extras_require={
        'test': ['nose', 'pytest', 'pytest-cov', 'codecov', 'coverage'],
    },
    test_suite='nose.collector'
)

