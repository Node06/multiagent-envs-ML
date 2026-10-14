from setuptools import setup, find_packages

setup(name='multiagent',
      version='0.0.1',
      description='Multi-Agent Goal-Driven Communication Environment',
      url='https://github.com/openai/multiagent-public',
      author='Igor Mordatch',
      author_email='mordatch@openai.com',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gymnasium==0.29.1', 'numpy==1.26.4',
                        'pyglet==1.5.24', 'six>=1.16,<2']
)
