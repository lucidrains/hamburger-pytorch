from setuptools import setup, find_packages

setup(
  name = 'hamburger-pytorch',
  packages = find_packages(),
  version = '0.0.3',
  license='MIT',
  description = 'Hamburger - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/hamburger-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'matrix factorization'
  ],
  install_requires=[
    'torch',
    'einops>=0.3'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)