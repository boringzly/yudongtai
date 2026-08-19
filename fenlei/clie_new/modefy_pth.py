import torch
import sys
import os
import json

new_dict = {'foreground_idx': 6}

def run():
    pth_path, dst_path = process_arguments(sys.argv)

    model_dict = torch.load(pth_path)
    model_dict.update(new_dict)
    torch.save(model_dict, dst_path)

def process_arguments(argv):
    if len(argv) < 3:
        help()
    pth_path = argv[1]
    dst_path = argv[2]

    return pth_path, dst_path

def help():
    print('Usage: python convert_pth.py pth_path dst_path')
    exit()

if __name__ == '__main__':
    run()