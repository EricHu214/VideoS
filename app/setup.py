# run python setup.py install

from setuptools import find_packages, setup

setup(
    name='VideoS',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'Flask>=1.1.2',
        'numpy>= 1.11',
        'opencv-python>=4.2.0.34',
        'PyInquirer>=1.0.3',
        'python-dotenv>=0.13.0',
        'tqdm'
    ],
)
