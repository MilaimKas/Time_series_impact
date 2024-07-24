from setuptools import setup

setup(name='TimeSeries_impact',
      version='0.1',
      description='Set of functions for advanced time series analysis and causal impact simulations',
      url='',
      author='Milaim Kas',
      author_email='milaim.kas@gmail.com',
      license='MIT',
      packages=['TimeSeries_impact'],
      install_requires=[
          'numpy', 'matplotlib', 'statsmodels', 'pandas', 'scipy', 'tdqm', 'tfcausalimpact'
      ],
      zip_safe=False)