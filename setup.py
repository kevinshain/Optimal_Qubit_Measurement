from setuptools import setup

setup(name='model',
      version='0.1',
      description='Optimized Qubit Parameter Estimation Model',
      url='https://github.com/p201-sp2016/Optimal_Qubit_Measurement',
      author='Kevin Shain',
      author_email='kshain@g.harvard.com',
      license='Kevin Shain',
      packages=['model'],
      install_requires=[
          'numpy',
      ],
      zip_safe=False)