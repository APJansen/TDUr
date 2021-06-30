from setuptools import setup

setup(
    name='td_ur',
    version='1.0',
    packages=['td_ur'],
    url='',
    license='MIT',
    author='Aron Jansen',
    author_email='',
    description='AI agent playing the Royal game of Ur',
    include_package_data=True,
    install_requires=['numpy', 'matplotlib', 'jax', 'jaxlib', 'ipywidgets']
)
