from change_detection_core import entry


def parse_bool(value):
    """兼容工作流通过命令行传入 true/false、1/0 等形式。"""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off', ''}:
        return False
    raise ValueError(f'无法解析布尔值: {value}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='变化检测 - 前后时相比较检测变化区域')
    parser.add_argument('--pre_image', type=str, required=True, help='前时相影像路径')
    parser.add_argument('--post_image', type=str, required=True, help='后时相影像路径')
    parser.add_argument('--model_path', type=str, required=False, default='./FBCD_test_select0207_best_acc.pth', help='主模型权重路径')
    parser.add_argument(
        '--use_two_models',
        type=parse_bool,
        nargs='?',
        const=True,
        default=False,
        help='是否启用两个变化检测权重并对结果做 OR 融合，默认 false',
    )
    parser.add_argument(
        '--second_model_path',
        type=str,
        required=False,
        default='./FBCD_test_Levir_CD_best_acc.pth',
        help='双模型模式下的第二个模型权重路径',
    )
    parser.add_argument('--dst_path', type=str, required=True, help='输出目录路径')
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
    model_path = args.model_path
    use_two_models = args.use_two_models
    second_model_path = args.second_model_path
    dst_path = args.dst_path
    output_dataset = args.output_dataset
    step_id = args.step_id
    step_name = args.step_name
    kafka_server_ip_port = args.kafka_server_ip_port
    kafka_topic = args.kafka_topic
    kafka_task_id = args.kafka_task_id
    debug = args.debug
    print(args)
    if debug: import pudb; pu.db
    entry(
        pre_image,
        post_image,
        model_path,
        dst_path,
        output_dataset,
        step_id,
        step_name,
        kafka_server_ip_port,
        kafka_topic,
        kafka_task_id,
        use_two_models=use_two_models,
        second_model_path=second_model_path,
    )
