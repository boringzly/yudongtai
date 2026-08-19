import torch as t


_name = 'CLInferEngine'
_version = '2.0.5'
_author = 'Cangling AI Team'


def info():
    print('{} {}'.format(_name, _version))
    print('Author: {}'.format(_author))
    print('PyTorch Version {}, {} Version {}'.format(t.__version__, _name, _version))