from setuptools import setup

setup(
    name='td_ur',
    version='1.0',
    packages=['td_ur'],
    package_data={'': ['*.pkl']},
    include_package_data=True,
    url='www.github.com/apjansen/TDUr',
    license='MIT',
    author='Aron Jansen',
    author_email='',
    description='AI agent playing the Royal game of Ur',
    install_requires=['numpy', 'matplotlib', 'jax', 'jaxlib', 'ipywidgets']
)
