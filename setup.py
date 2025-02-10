from setuptools import setup, find_packages

setup(
    name='r1helper',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your dependencies here
        # e.g., 'numpy>=1.18.0',
        # e.g., 'pandas',
    ],
    author='Chris Wendler',
    author_email='chris.wendler.mobile@gmail.com',
    description='Helper functions for mechanistic interpretability research with R1.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/wendlerc/r1helpers',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)