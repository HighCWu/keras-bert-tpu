from setuptools import setup, find_packages

setup(
    name='keras-bert-tpu',
    version='0.1.7',
    packages=find_packages(),
    url='https://github.com/HighCWu/keras-bert-tpu',
    license='MIT',
    author='HighCWu',
    author_email='HighCWu@163.com',
    description='BERT implemented in Keras of Tensorflow package on TPU',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy>=1.15.4',
        'tensorflow>1.12,<2.0',
    ],
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
