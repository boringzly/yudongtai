from classification_core import entry

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='变化检测类别识别 - 对变化区域进行前后时相地类分类')
    parser.add_argument('--pre_image', type=str, required=True, help='前时相影像路径')
    parser.add_argument('--post_image', type=str, required=True, help='后时相影像路径')
    parser.add_argument('--mask_shp', type=str, required=True, help='变化检测结果SHP路径')
    parser.add_argument('--model_path', type=str, required=False, default='./clie_new/tools/fullclass_chinaall_3b.pth', help='分类模型路径')
    parser.add_argument('--dst_path', type=str, required=True, help='输出目录路径')
    parser.add_argument('--path_working', type=str, required=False, default=None, help='工作目录路径，用于存放中间SHP和PNG')
    parser.add_argument('--output_dataset', type=str, required=False, default=None)
    parser.add_argument('--step_id', type=str, required=False, default=None)
    parser.add_argument('--step_name', type=str, required=False, default=None)
    parser.add_argument('--kafka_server_ip_port', type=str, required=False, default='')
    parser.add_argument('--kafka_topic', type=str, required=False, default='')
    parser.add_argument('--kafka_task_id', type=str, required=False, default=None)
    parser.add_argument('--debug', action='store_true')
    args, unknown = parser.parse_known_args()
    pre_image = args.pre_image
    post_image = args.post_image
    mask_shp = args.mask_shp
    model_path = args.model_path
    dst_path = args.dst_path
    path_working = args.path_working
    output_dataset = args.output_dataset
    step_id = args.step_id
    step_name = args.step_name
    kafka_server_ip_port = args.kafka_server_ip_port
    kafka_topic = args.kafka_topic
    kafka_task_id = args.kafka_task_id
    debug = args.debug
    print(args)
    if debug: import pudb; pu.db
    entry(pre_image, post_image, mask_shp, model_path, dst_path, path_working, output_dataset, step_id, step_name,
          kafka_server_ip_port, kafka_topic, kafka_task_id)
