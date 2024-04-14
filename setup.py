from setuptools import setup, find_packages

setup(
    name='explainablefl',
    version='0.1.0',
    author='Ratheshan Sathiyamoorthy',
    author_email='lionratheshan@gmail.com',
    packages=find_packages(),
    install_requires=[
        'shap>=0.39.0',  
        'matplotlib>=3.0', 
        'torch>=1.8',
        'numpy',
    ],
    description='A package for explaining federated learning models using SHAP values.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    keywords='federated learning explainable AI SHAP',
    url='https://github.com/ratheshan03/explainablefl',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
