import warnings


class BaseInferConfig(object):
    load_model_path = './tools/best_model.pth'
    resume = 0

    test_img_root = './data/default/'
    test_img_suffix = '.tif'
    test_img_list_file = None
    test_output_root = './data/default/'
    gdal_driver = 'GTiff'
    tiff_compress = 'LZW'
    mean_file = './tools/mean_value.txt'
    std_file = './tools/std_value.txt'
    band_list_file = './tools/band_list.txt'
    color_table_file = './tools/color_table.txt'

    retry = 0
    error_log = None

    models_version = 'v1'

    # models_v2
    custom_model = False
    model = 'RSEffUNet'
    encoder_name = 'resnet34'
    encoder_depth = 5
    encoder_dilation = True
    encoder_output_stride = 16

    decoder_use_batchnorm = True
    decoder_channels = (256, 128, 64, 32, 16)
    decoder_attention_type = None
    decoder_pyramid_channels = 256
    decoder_segmentation_channels = 128
    decoder_merge_policy = 'add'
    decoder_atrous_rates = (12, 24, 36)
    decoder_dropout = 0.2
    decoder_pab_channels = 64
    psp_out_channels = 512
    psp_use_batchnorm = True
    decoder_res_type = 'basicblock'

    # models_v1
    basic_net = 'efficientnet-b3'
    norm_name = 'BatchNorm2d'
    psp_sizes = (1, 2, 3, 6)
    drop_rate = 0

    inchannel = 4
    num_classes = 14
    activation = None
    aux_params = None
    upsampling = 8
    
    batch_size = 8
    img_size = 1024
    pixel_overlap = 64
    force_to_single_class = 10
    use_binary = True
    binary_threshold = 0.5

    use_gpu = True
    gpu_ids = [0]
    num_workers = 4
    use_cudnn_benchmark = True

    use_tta = False
    tta_flip = False
    tta_rotate = False

    pin_memory = True
    non_blocking = True
    use_dist = True

    debug_file = './runs/debug'

    exception_value = None
    use_full_out = False
    use_single_out = False
    use_color_out = True
    use_shapefile_out = True
    use_vec_temp_out = True
    use_output_root_reset = False
    use_transparent_background = False


def parse(self, kwargs):
    for k, v in kwargs.items():
        if k == 'custom_config':
            continue
        if not hasattr(self, k):
            warnings.warn('Warnings: opt has not attribute %s' % k)
        setattr(self, k, v)


BaseInferConfig.parse = parse
bareland_opt = BaseInferConfig()