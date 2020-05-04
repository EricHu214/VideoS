# run python setup.py install

from setuptools import find_packages, setup

setup(
    name='VideoS',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'Flask-Migrate>=2.5.3',
        'Flask-SQLAlchemy>=2.4.1',
        'Flask>=1.1.2',
        'PyInquirer>=1.0.3',
        'psycopg2>=2.8.5',
        'python-dotenv>=0.13.0',
        'numpy',
        'opencv-python',
    ],
)
