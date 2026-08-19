import torch
import sys
import os
import json

target_dict = {
    "color_table": "/nfs/project/netdisk/100/workspace/change_detection/private/chenpan/tools/color_table_14class.txt",
    "mean_value": "/nfs/project/netdisk/192.168.10.226/d/private/wangzq/2022.2.14-fullclass_14w/mean_value.txt",
    "std_value": "/nfs/project/netdisk/192.168.10.226/d/private/wangzq/2022.2.14-fullclass_14w/std_value.txt",
    "model_num_classes": 1,
    "threshold": 0.5,
    "img_size": 1536,
    "band_num": 3,
    "model_output_stride": 32,
    "foreground_idx": 11,
    "model_name": "SwinUNet",
    "model_backbone": "Swin_tiny_p4w7",
    "description": {
            "Name": "内蒙水体微调模型",
            "Version": "V1.0",
            "UpdateTime": "20221018",
            "Resolution": "1-2m",
            "Band": "3波段",
            "Describe": "适用于3波段特定内蒙影像水体提取模型"
        }

}

def recurent_find_pth(inpath, pth_list):
    contents = os.listdir(inpath)
    for content in contents:
        if os.path.isdir(os.path.join(inpath, content)):
            recurent_find_pth(os.path.join(inpath, content), pth_list)
        else:
            if "allinone" in content:
                continue
            if content[-4:] == '.pth':
                pth_list.append(os.path.join(inpath, content))

def read_mean_std(file_path):
    values = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line != "":
                values.append(float(line))
    return values

def generate_color_table(color_table_file):
    color_table = []
    with open(color_table_file, 'r') as f:
        for line in f:
            color_table.append(tuple([int(i) for i in line.strip().split('#')[0].split('/')]))
    return color_table

def load_discribe_file(jsonfile):
    with open(jsonfile, 'r') as f:
        return json.load(f)

def run():
    src_pth, dst_pth = process_arguments(sys.argv)
    
    model_dict = torch.load(src_pth)
    for key, value in target_dict.items():
        if key == "mean_value":
            # import ipdb;ipdb.set_trace()
            mean_value = read_mean_std(value)
            model_dict["mean_value"] = mean_value
        elif key == "std_value":
            std_value = read_mean_std(value)
            model_dict["std_value"] = std_value
        elif key == "color_table":
            color_table = generate_color_table(value)
            model_dict["color_table"] = color_table
        else:
            model_dict[key] = value
            
        torch.save(model_dict, dst_pth)

def process_arguments(argv):
    if len(argv) < 3:
        help()
    src_pth = argv[1]
    dst_pth = argv[2]

    return src_pth, dst_pth

def help():
    print('Usage: python convert_pth.py src_pth dst_pth')
    exit()

if __name__ == '__main__':
    run()