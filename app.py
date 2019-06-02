import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_inference
import mnist_train


def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32,[1,mnist_inference.IMAGE_SIZE,
                                        mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS])
        y = mnist_inference.inference(x,False,None)
        preValue = tf.argmax(y,1)

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)

                preValue = sess.run(preValue,feed_dict={x:testPicArr})
                return preValue
            else:
                print("No checkpoint file found!")
                return -1


def pre_img(PicName):
    img = Image.open(PicName)
    reIm = img.resize((28,28),Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255-im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1,784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr,1.0/255.0)
    img_ready1 = np.reshape(img_ready,(1,mnist_inference.IMAGE_SIZE,
                                        mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS))

    return img_ready1

def app_run():
    testNum = int(input('输入要测试的图片数量:'))
    for i in range(testNum):
        testPic = input("输入图片路径:")
        true_val = testPic.split('/')[-1].split('.')[-2]
        testPicArr = pre_img(testPic)
        prevalue = restore_model(testPicArr)
        #print(true_val,int(prevalue))
        if int(true_val) == int(prevalue):
            flg = '识别正确！'
        else:
            flg = '识别错误！'
        #print(flg)
        print('识别后的输出是:',int(prevalue) , ','+ flg)

def main():
    app_run()

if __name__=='__main__':
    main()
