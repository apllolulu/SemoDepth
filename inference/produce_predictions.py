import tensorflow as tf
import inputs as inputs
import inference_ResNet_ls_eval as inference
import scipy.io as sio
import config as cfg
import numpy as np
from PIL import Image
import cv2
from numpy import mat

def predict(config):
    with tf.Graph().as_default():

        data_iterator, iterator_init_op = inputs.generate_data_iterator(cfg.FLAGS.filenames_path)
        next_img, next_savepath = data_iterator.get_next()
        shape = tf.shape(next_img)
        
        #print("next_img.shape:",next_img.shape) #(?, 188, 621, 3)
     
        prediction = inference.inference(next_img)
        prediction = cfg.FLAGS.bf / prediction

        saver = tf.train.Saver(tf.global_variables())
 
        sess = tf.Session(config=config)
        sess.run([tf.global_variables_initializer(), iterator_init_op])

        
        saver.restore(sess, cfg.FLAGS.chkpt_path)

        while True:
            try:
                
                _prediction, _savename, _sh = sess.run([prediction, next_savepath, shape])                            
                data = np.matrix(_prediction[0])
               

                #cv2.imshow("depth:",data)
                im = Image.fromarray(data)
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                    im = im.resize((next_img.shape[2], next_img.shape[1]),Image.ANTIALIAS)
                im.show()
                #print("im.shape",im.shape)
                #im = mat(im)
                #cv2.imshow("depth:",im)
            #    im.save("depth.png")
                

            #    savename = _savename[0]
            #    sio.savemat(savename, {"mat":_prediction})
            except Exception as e:
               
                if e.message.find("End of sequence") != -1:
                    print("Finished")
                else:
                    print(str(e))
                break


def test():
    with tf.Graph().as_default():
        
        config = tf.ConfigProto(
            device_count={'GPU': 1}
        )
        config.gpu_options.allow_growth = False
        config.gpu_options.allocator_type = 'BFC'


        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
   
        delay=60/int(fps)
    
        while(cap.isOpened()):
            ret,frame = cap.read()
            if ret==True:
                image = frame   
                image = image[:,:672]
                cv2.imshow('image', image)##376 672 3
                cv2.imwrite('001.png', image)

                data_iterator, iterator_init_op = inputs.generate_data_iterator(cfg.FLAGS.filenames_path)
                next_img, next_savepath = data_iterator.get_next()
                shape = tf.shape(next_img)
                prediction = inference.inference(next_img)
                prediction = cfg.FLAGS.bf / prediction

                saver = tf.train.Saver(tf.global_variables())
                sess = tf.Session(config=config)
                sess.run([tf.global_variables_initializer(), iterator_init_op])
                saver.restore(sess, cfg.FLAGS.chkpt_path)

                _prediction, _savename, _sh = sess.run([prediction, next_savepath, shape])                            
                data = np.matrix(_prediction[0])
               
                im = Image.fromarray(data)
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                    im = im.resize((next_img.shape[2], next_img.shape[1]),Image.ANTIALIAS)
             
                im.show()


                cv2.waitKey(np.uint(delay))
            
            else:
                break
                cap.release()
                cv2.destroyAllWindows()

       

def main(argv=None):
    config = tf.ConfigProto(
            device_count={'GPU': 1}
        )
    config.gpu_options.allow_growth = False
    config.gpu_options.allocator_type = 'BFC'

   
    
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
   
    delay=60/int(fps)
    
    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret==True:
            image = frame   

            image = image[:,:672]
            cv2.imshow('image', image)##376 672 3
            cv2.imwrite('001.png', image)

            predict(config)
   
            cv2.waitKey(np.uint(delay))
            
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    tf.app.run()




