"""
深度预测
"""
import tensorflow as tf

# 卷积层
def conv(input, shape, stride = [1, 1, 1, 1], use_bias = False, kv_name = "weights", bv_name = "bias"):
    
    sh_in = input.get_shape().as_list()
    
    shape = [shape[0], shape[1], sh_in[3], shape[2]]
    kernel = tf.get_variable(kv_name, shape, dtype=tf.float32)
    conv = tf.nn.conv2d(input, kernel, stride, padding='SAME')
    if use_bias:
        bias = tf.get_variable(bv_name, [shape[3]], dtype=tf.float32)
        conv = tf.nn.bias_add(conv, bias)

    return conv

 
# 批标准化
def bn(input, sv_name = "scale", ov_name = "offset", mv_name = "mean", vv_name = "variance"):
    
    shape = input.get_shape().as_list()
    n_channels = shape[3]
    scale = tf.get_variable(sv_name, [n_channels], dtype=tf.float32)
    offset = tf.get_variable(ov_name, [n_channels], dtype=tf.float32)
    mean = tf.get_variable(mv_name, [n_channels], dtype=tf.float32)
    variance = tf.get_variable(vv_name, [n_channels], dtype=tf.float32)
    out = tf.nn.batch_normalization(input, mean, variance, offset, scale, 0.00000001)

    return out

# 特征连接
def concat_pad(upprojection, res_block):
    
    shape1 = res_block.get_shape().as_list()[1:-1]
    shape2 = upprojection.get_shape().as_list()[1:-1]
    padding = [a_i - b_i for a_i, b_i in zip(shape2, shape1)]
    block_padded = tf.pad(res_block, [[0, 0], [0, padding[0]], [0, padding[1]], [0, 0]])
    res = tf.concat([upprojection, block_padded], 3)
    return res

# 快速上卷积层
def fast_upconv(input, n_out_channels):
    """
    输入：
    input: 输入特征图的四维张量 [batch_size, height, width, depth]
    n_out_channels:输出特征图的深度

    返回:
    res: 四维张量 [batch_size, height * 2, width * 2, n_out_channels]
   
    """
    
    conv1 = conv(input, [3, 3, n_out_channels], use_bias=True, kv_name="weights1", bv_name="bias1")
    conv2 = conv(input, [2, 3, n_out_channels], use_bias=True, kv_name="weights2", bv_name="bias2")
    conv3 = conv(input, [3, 2, n_out_channels], use_bias=True, kv_name="weights3", bv_name="bias3")
    conv4 = conv(input, [2, 2, n_out_channels], use_bias=True, kv_name="weights4", bv_name="bias4")

    # 连接卷积输出
    sh = conv1.get_shape().as_list()
    dim = len(sh[1:-1])
    tmp1 = tf.reshape(conv1, [-1] + sh[-dim:])
    tmp2 = tf.reshape(conv2, [-1] + sh[-dim:])
    tmp3 = tf.reshape(conv3, [-1] + sh[-dim:])
    tmp4 = tf.reshape(conv4, [-1] + sh[-dim:])

    # 水平方向
    concat1 = tf.concat([tmp1, tmp3], 2)
    concat2 = tf.concat([tmp2, tmp4], 2)

    # 垂直方向
    concat_final = tf.concat([concat1, concat2], 1)
    out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
    res = tf.reshape(concat_final, out_size)

    return res

# 快速向上投影层
def fast_upproject(input, n_out_channels, variable_scope):
    """
    输入：
    input: 输入特征图的四维张量 [batch_size, height, width, depth]
    n_out_channels: 输出特征图的深度

    返回:
    res:四维张量 [batch_size, height * 2, width * 2, n_out_channels]
   
    """
    with tf.variable_scope(variable_scope):
        
        with tf.variable_scope('fast1_c'):
            conv1 = fast_upconv(input, n_out_channels)
        conv_bn1 = tf.nn.relu(bn(conv1, vv_name="variance1", ov_name="batchnorm_offset1", mv_name="mean1", sv_name="batchnorm_scale1"))

        
        conv2 = conv(conv_bn1, [3, 3, n_out_channels], kv_name="weights2")
        conv_bn2 = bn(conv2, vv_name="variance2", ov_name="batchnorm_offset2", mv_name="mean2", sv_name="batchnorm_scale2")

        
        with tf.variable_scope('fast2_c'):
            conv3 = fast_upconv(input, n_out_channels)
        conv_bn3 = bn(conv3, vv_name="variance3", ov_name="batchnorm_offset3", mv_name="mean3", sv_name="batchnorm_scale3")

        # 求和
        sum = conv_bn2 + conv_bn3

        res = tf.nn.relu(sum)

    return res

# 应用残差
def res_block(input, nfilter1, nfilter2, use_shortcut=False, stride=1, variable_scope=""):
    """
    输入:
    input: 输入特征图的四维张量 [batch_size, height, width, depth]
    nfilter1: 第一第二卷积核的数量.
    nfilter2: 第三层卷积核的数量.

    返回:
    res: 四维张量 [batch_size, height, width, nfilter2]
   
    """
    with tf.variable_scope(variable_scope):
        
        conv1 = conv(input, [1, 1, nfilter1], [1, stride, stride, 1], kv_name="weights1")
        conv1_bn = bn(conv1, vv_name="variance1", ov_name="batchnorm_offset1", mv_name="mean1", sv_name="batchnorm_scale1")
        conv1_relu = tf.nn.relu(conv1_bn)

        
        conv2 = conv(conv1_relu, [3, 3, nfilter1], kv_name="weights2")
        conv2_bn = bn(conv2, vv_name="variance2", ov_name="batchnorm_offset2", mv_name="mean2", sv_name="batchnorm_scale2")
        conv2_relu = tf.nn.relu(conv2_bn)

        
        conv3 = conv(conv2_relu, [1, 1, nfilter2], kv_name="weights3")
        conv3_bn = bn(conv3, vv_name="variance3", ov_name="batchnorm_offset3", mv_name="mean3", sv_name="batchnorm_scale3")

        if use_shortcut:
            
            conv4 = conv(input, [1, 1, nfilter2], [1, stride, stride, 1], kv_name="weights4")
            conv4_bn = bn(conv4, vv_name="variance4", ov_name="batchnorm_offset4", mv_name="mean4", sv_name="batchnorm_scale4")

           
            res = tf.nn.relu(tf.add(conv4_bn, conv3_bn))
        else:
            res = tf.nn.relu(tf.add(input, conv3_bn))

    return res


# 使用快速向上投影和长跳层连接 编码解码 ResNet-50 结构
def inference(images):
    """
    输入：
    images: 输入图片的四维张量 [batch_size, image_height, image_width, num_channels]
    返回:
    prediction: 预测视差图的四维张量 [batch_size, image_height / 2, image_width / 2, 1] 
    """
   
    # 编码部分
    # 第一卷积层
    with tf.variable_scope("conv1"):
        conv1 = conv(images, [7, 7, 64], [1, 2, 2, 1])
        conv1 = bn(conv1, vv_name="variance", ov_name="batchnorm_offset", mv_name="mean", sv_name="batchnorm_scale")
        conv1 = tf.nn.relu(conv1)

    # 第一池化层
    pool1 = tf.nn.max_pool(conv1, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')

    # 残差单元
    block1 = res_block(pool1, 64, 256, True, 1, 'block1')
    block2 = res_block(block1, 64, 256, variable_scope='block2')
    block3 = res_block(block2, 64, 256, variable_scope='block3')

    # Scale x2
    block4 = res_block(block3, 128, 512, True, 2, 'block4')
    block5 = res_block(block4, 128, 512, variable_scope='block5')
    block6 = res_block(block5, 128, 512, variable_scope='block6')
    block7 = res_block(block6, 128, 512, variable_scope='block7')

    # Scale x2
    block8 = res_block(block7, 256, 1024, True, 2, 'block8')
    block9 = res_block(block8, 256, 1024, variable_scope='block9')
    block10 = res_block(block9, 256, 1024, variable_scope='block10')
    block11 = res_block(block10, 256, 1024, variable_scope='block11')
    block12 = res_block(block11, 256, 1024, variable_scope='block12')
    block13 = res_block(block12, 256, 1024, variable_scope='block13')

    # Scale x2
    block14 = res_block(block13, 512, 2048, True, 2, 'block14')
    block15 = res_block(block14, 512, 2048, variable_scope='block15')
    block16 = res_block(block15, 512, 2048, variable_scope='block16')
   

    # 第二卷积层
    with tf.variable_scope("conv2"):
        conv2 = conv(block16, [1, 1, 1024])
        conv2 = bn(conv2, vv_name="variance", ov_name="batchnorm_offset", mv_name="mean", sv_name="batchnorm_scale")
    # 结束编码

    # 开始解码
    upproject1 = fast_upproject(conv2, 512, 'upproject1')
    upproject1 = concat_pad(upproject1, block13)

    upproject2 = fast_upproject(upproject1, 256, 'upproject2')
    upproject2 = concat_pad(upproject2, block7)

    upproject3 = fast_upproject(upproject2, 128, 'upproject3')
    upproject3 = concat_pad(upproject3, block3)

    upproject4 = fast_upproject(upproject3, 64, 'upproject4')

    # 最后一个卷积来调整预测深度
    with tf.variable_scope("conv3"):
        prediction = conv(upproject4, [3, 3, 1], use_bias = True)

    return prediction
