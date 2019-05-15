import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('input_image_channels', 3, """Number of channels in the input image.""")
tf.app.flags.DEFINE_integer('inference_image_height', 188, """Height of the image for inference.""") #188
tf.app.flags.DEFINE_integer('inference_image_width', 621, """Width of the image for inference.""") #621

tf.app.flags.DEFINE_string('filenames_path', "file.txt", """Path to file contatining input-output names.""")
tf.app.flags.DEFINE_string('chkpt_path', "./model/depth-chkpt", """Path to saved model without extension""")

tf.app.flags.DEFINE_float('bf', 359.7176277195809831 * 0.54, """Baseline times focal length""")
