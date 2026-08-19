import warnings


class InferConfig(object):
    custom_config = None
    load_model_path = None
    resume = 0

    test_img_root = './data/default/'
    test_img_suffix = '.tif'
    test_img_list_file = None
    test_output_root = './data/default/'
    gdal_driver = 'GTiff'
    tiff_compress = 'LZW'
    mean_file = './data/default/mean_value.txt'
    std_file = './data/default/std_value.txt'
    band_list_file = './data/default/band_list.txt'
    color_table_file = './data/default/color_table.txt'

    retry = 0
    error_log = None

    models_version = 'v2'

    # models_v2
    custom_model = False
    model = 'Unet'
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
    basic_net = 'resnet50'
    norm_name = 'BatchNorm2d'
    psp_sizes = (1, 2, 3, 6)
    drop_rate = 0.1

    inchannel = 3
    num_classes = 2
    activation = None
    aux_params = None
    upsampling = 8
    
    batch_size = 1
    img_size = 512
    pixel_overlap = 256
    force_to_single_class = None
    use_binary = True
    binary_threshold = 0.5

    use_gpu = True
    gpu_ids = [0]
    num_workers = 0
    use_cudnn_benchmark = True

    use_tta = False
    tta_flip = False
    tta_rotate = False

    pin_memory = False
    non_blocking = False
    use_dist = False

    debug_file = './runs/debug'

    exception_value = None
    use_full_out = False
    use_single_out = False
    use_color_out = False
    use_shapefile_out = False
    use_output_root_reset = False
    use_transparent_background = False


def parse(self, kwargs):
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warnings: opt has not attribute %s" % k)
        setattr(self, k, v)


def user(self):
    print('user config:')
    config_dict = self.__class__.__dict__.items()
    config_dict_ordered = sorted(config_dict, key=lambda d:d[0])
    for k, v in config_dict_ordered:
        if not k.startswith('__'):
            print((k, getattr(self, k)))


def load_custom_config(self, custom_opt):
    custom_opt_dict = custom_opt.__class__.__dict__.items()
    for k, v in custom_opt_dict:
        if not k.startswith('__'):
            if not hasattr(self, k):
                warnings.warn('Warnings: opt has not attribute %s' % k)
            setattr(self, k, v)
    custom_opt_dict = custom_opt.__dict__.items()
    for k, v in custom_opt_dict:
        if not k.startswith('__'):
            if not hasattr(self, k):
                warnings.warn('Warnings: opt has not attribute %s' % k)
            setattr(self, k, v)


InferConfig.parse = parse
InferConfig.user = user
InferConfig.load_custom_config = load_custom_config
opt = InferConfig()