import sys
from CLInferEngine.run import infer
import fire

def run(**kwargs):
    infer(kwargs)

def help():
    print("""
    usage : python run.py <function> [--args=value]
    <function> := infer | help
    example:
            python {0} infer --test-img-root='data/default/'
            python {0} help
    available args:
    """.format(__file__))

    source = (getsource(_opt.__class__))
    print(source)

if __name__ == '__main__':
    fire.Fire()