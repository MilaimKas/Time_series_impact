from setuptools import setup, find_packages

setup(name='TimeSeries_impact',
      version='0.3',
      description='Set of functions for advanced time series analysis and causal impact simulations',
      url='',
      author='Milaim Kas',
      author_email='milaim.kas@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy', 'matplotlib', 'statsmodels', 'pandas', 'scipy', 'tdqm', 'tfcausalimpact', 'seaborn', 'pytensor'
      ],
      zip_safe=False)