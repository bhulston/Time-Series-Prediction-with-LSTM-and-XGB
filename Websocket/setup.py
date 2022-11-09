from setuptools import setup
from os.path import join, dirname

here = dirname(__file__)

setup(name='bitmex-ws',
      version='0.5.0',
      description='Adapter for connecting to the BitMEX Websocket API.',
      #long_description=open(join(here, 'README.md')).read(),
      author='Brandon Hulston',
      author_email='hulston@usc.edu',
      url='',
      install_requires=[
          'websocket-client==0.53.0',
      ],
      packages=['', 'util'],
      )
