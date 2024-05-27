from setuptools import setup, find_packages

setup(
    name='psimol',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'ruamel.yaml',
    ],
    entry_points={
        'console_scripts': [
            'psimol=psimol.cli:main',
        ],
    },
    author='Dawid Uchal, Maciej Bielecki, Paweł Nagórko',
    description='Python plugin for psi4 package.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dav3794/PsiMol',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
