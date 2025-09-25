from setuptools import setup, find_packages

setup(
    name='streamlit_sales_predictor',
    version='0.1.0',
    author='',
    author_email='',
    description='A Streamlit app for sales prediction and insight generation',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'streamlit>=1.25.0',
        'pandas>=1.5.0',
        'numpy>=1.23.0',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.6.0',
        'plotly>=5.15.0',
        'altair>=5.0.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Framework :: Streamlit',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
