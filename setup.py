from setuptools import setup, find_packages

setup(
    name='semiconductor-defect-ml',
    version='0.1.0',
    author='Ashutosh Tiwari',
    author_email='d845112008@tmu.edu.tw',
    description='Physics-Informed Machine Learning for Semiconductor Defect Level Prediction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Ashu-design/semiconductor-defect-piml',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'shap'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
)
