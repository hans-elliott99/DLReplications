from setuptools import setup, find_namespace_packages

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='DeepLearningReplications',
    url='https://github.com/hans-elliott99/dl-replications',
    author='Hans Elliott',
    author_email='hanselliott61@gmail.com',
    # Needed to actually package something
    packages=find_namespace_packages(include=['dlrep.*']),
    # Needed for dependencies
    install_requires=['torch', 'torchvision', 'numpy', 'nltk', 'opencv_python_headless', 'Pillow', 'scikit_image'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='An example of a python package from pre-existing code',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
