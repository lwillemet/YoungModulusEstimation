from setuptools import setup

requirements = [
    'numpy',
    'numpy-quaternion',
    'opencv-python',
    # 'open3d',
    'autolab_core'
]

setup(name='gelsight',
      version='1.0.0',
      description='Gelsight sensor interface',
      author='Shaoxiong Wang',
      author_email='',
      package_dir = {'': '.'},
      packages=['gelsight'],
      install_requires = requirements,
      extras_require = {}
     )
