import tensorflow as tf
import config as cfg

# 读取输入图像并将输入文件名映射到它
def process_record(in_path, out_path):
    """
    输入:
    in_path
    out_path

    返回值:
    img: 输入图片的三维张量[cfg.FLAGS.inference_image_height, cfg.FLAGS.inference_image_width,cfg.FLAGS.inference_image_channels] 
    out_path
    """

    # 读入输入图片
    img_contents = tf.read_file(in_path)
    #print("img_contents.shape",img_contents.shape)
    # 将原始文件解码为png
    img = tf.cast(tf.image.decode_png(img_contents, channels=cfg.FLAGS.input_image_channels), dtype=tf.float32)
    
    # 调整图像大小以匹配单个管道输入大小
    img = tf.image.resize_images(img, [cfg.FLAGS.inference_image_height, cfg.FLAGS.inference_image_width],
                                 method=tf.image.ResizeMethod.BILINEAR)
    print("img.shape:",img.shape)
    return img, out_path

# 读取输入和输出文件名列表
def read_filenames(filenames_path):
    """
    filenames_path: 包含输入图像名称对和保存预测的文件名的文件的路径。
    返回:
    l_in: 输入图片列表.
    l_out: 输出名字列表.
    """

    f = open(filenames_path, 'r')
    l_in = []
    l_out = []
    for line in f:
        try:
            i_name, o_name = line[:-1].split(',')
            #print("i_name:",i_name)
            #print("o_name:",o_name)
        except ValueError:
            i_name = o_name = line.strip("\n")
            print("Something is wrong with the filelist")
        
        if not tf.gfile.Exists(i_name):
            raise ValueError('Failed to find file: ' + i_name)
        
        l_in.append(i_name)
        l_out.append(o_name)

    return l_in, l_out

# 生成数据迭代器
def generate_data_iterator(filenames_path):
    """
    输入:
    filenames_path

    返回:
    data_iterator
    iterator_init_op
    """
    
    img, savepath = read_filenames(filenames_path)

    #转化为张量
    img = tf.constant(img)
    generate_data_iterator = tf.constant(savepath)
    
    data = tf.data.Dataset.from_tensor_slices((img, savepath))
    
    data = data.map(process_record)
    
    
    data = data.batch(1)
    
    data_iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
    iterator_init_op = data_iterator.make_initializer(data)

    return data_iterator, iterator_init_op
