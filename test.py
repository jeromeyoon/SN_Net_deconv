import os
import time
from glob import glob
import tensorflow as tf

#from tensorflow.python.ops.script_ops import *
from ops import *
from utils import *

class EVAL(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=1, input_size=64, sample_size=32, ir_image_shape=[64, 64,1], normal_image_shape=[64, 64, 3],
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None):

        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.normal_image_shape = normal_image_shape
        self.ir_image_shape = ir_image_shape

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = 3
	
	self.g_bn1 = batch_norm(self.batch_size, name='g_bn1')
	self.g_bn2 = batch_norm(self.batch_size, name='g_bn2')
	self.g_bn3 = batch_norm(self.batch_size, name='g_bn3')
	self.g_bn4 = batch_norm(self.batch_size, name='g_bn4')
	self.g_bn5 = batch_norm(self.batch_size, name='g_bn5')
	self.d_normal_bn1 = batch_norm(self.batch_size, name='d_bn1')
        self.d_normal_bn2 = batch_norm(self.batch_size, name='d_bn2')
        self.d_normal_bn3 = batch_norm(self.batch_size, name='d_bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):

        self.ir_images = tf.placeholder(tf.float32, [self.batch_size] + self.ir_image_shape,
                                    name='ir_images')
        self.normal_images = tf.placeholder(tf.float32, [self.batch_size] + self.normal_image_shape,
                                    name='normal_images')

        self.G = self.generator(self.ir_images)
        self.sampler = self.sampler(self.ir_images)
        self.saver = tf.train.Saver()

    def generator(self, nir, y=None):
	#tf.get_variable_scope().reuse_variables()
	h0 =tf.nn.relu(self.g_bn1(conv2d(nir,self.df_dim,name='g_nir_1')))
        h1 =tf.nn.relu(self.g_bn2(conv2d(h0,20,k_h=1,k_w=1,name='g_nir_2')))
        h2 =tf.nn.relu(self.g_bn3(conv2d(h1,20,k_h=3,k_w=3,name='g_nir_3_1')))
        h3 =tf.nn.relu(self.g_bn4(conv2d(h2,20,k_h=3,k_w=3,name='g_nir_3_2')))
        #h4 =lrelu(conv2d(h3,20,k_h=3,k_w=3,name='g_nir_3_3'))
        #h5 =lrelu(conv2d(h4,20,k_h=3,k_w=3,name='g_nir_3_4'))
        #h6 =lrelu(conv2d(h5,20,k_h=3,k_w=3,name='g_nir_3_5'))
	h4 =tf.nn.relu(self.g_bn5(conv2d(h3,self.df_dim,k_h=1,k_w=1,name='g_nir_4')))
        h5 =deconv2d(h4,[self.batch_size,600,800,3],name='g_nir_5',with_w=False)
	return tf.nn.tanh(h5)
    
    def sampler(self, nir, y=None):
	tf.get_variable_scope().reuse_variables()
	h0 =tf.nn.relu(self.g_bn1(conv2d(nir,self.df_dim,name='g_nir_1'),train=False))
        h1 =tf.nn.relu(self.g_bn2(conv2d(h0,20,k_h=1,k_w=1,name='g_nir_2'),train=False))
        h2 =tf.nn.relu(self.g_bn3(conv2d(h1,20,k_h=3,k_w=3,name='g_nir_3_1'),train=False))
        h3 =tf.nn.relu(self.g_bn4(conv2d(h2,20,k_h=3,k_w=3,name='g_nir_3_2'),train=False))
        #h4 =lrelu(conv2d(h3,20,k_h=3,k_w=3,name='g_nir_3_3'))
        #h5 =lrelu(conv2d(h4,20,k_h=3,k_w=3,name='g_nir_3_4'))
        #h6 =lrelu(conv2d(h5,20,k_h=3,k_w=3,name='g_nir_3_5'))
	h4 =tf.nn.relu(self.g_bn5(conv2d(h3,self.df_dim,k_h=1,k_w=1,name='g_nir_4'),train=False))
        h5 =deconv2d(h4,[self.batch_size,600,800,3],name='g_nir_5',with_w=False)
	return tf.nn.tanh(h5)


    def load(self, checkpoint_dir,model):
        print(" [*] Reading checkpoints...")

        #model_dir = "%s_%s" % (self.dataset_name, 32)
        model_dir = "%s" % (self.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
	#model_path = os.path.join(checkpoint_dir,model)
	if os.path.isfile(os.path.join(checkpoint_dir,model)):
	    print(' Success load network ')
	    self.saver.restore(self.sess, os.path.join(checkpoint_dir, model))
	    return True
	else:
	    print('Fail to load network')
	    return False
	"""
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print('*************** ckpt *************')
        print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.all_model_checkpoint_paths[-3])
            print('Loaded network:',ckpt_name)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
	"""
             
