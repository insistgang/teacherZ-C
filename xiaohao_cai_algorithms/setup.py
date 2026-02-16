from setuptools import setup, find_packages

setup(
    name='xiaohao_cai_algorithms',
    version='1.0.0',
    description='Image analysis algorithms based on Xiaohao Cai research',
    author='Xiaohao Cai',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20',
        'scipy>=1.7',
        'opencv-python>=4.5',
        'torch>=1.10',
        'tensorly>=0.7',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
