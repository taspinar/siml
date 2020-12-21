from setuptools import setup

setup(
		name='siml',
		version='0.4.0',
		description='Machine Learning algorithms implemented from scratch',
		url='https://github.com/taspinar/siml',
		author='Ahmet Taspinar',
		author_email='taspinar@gmail.com',
		license='MIT',
		packages=['siml'],
		install_requires=['numpy', 'scikit-learn', 'pandas','matplotlib'],
		zip_safe=False
		)
