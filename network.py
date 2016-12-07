from ops import *
import tensorflow as tf

class networks(object):
    def __init__(self,num_block,batch_size,df_dim):
	self.num_block = num_block
	self.batch_size = batch_size
	self.df_dim = df_dim
    def generator(self,nir):
        g_nir0 =tf.nn.relu(conv2d(nir,self.df_dim,name='g_nir0'))
	g_bn1 = batch_norm(self.batch_size,name='g_bn1')
        g_nir1 =tf.nn.relu(g_bn1(conv2d(g_nir0,self.df_dim*8,name='g_nir1')))
	g_bn2 = batch_norm(self.batch_size,name='g_bn2')
        g_nir2 =tf.nn.relu(g_bn2(conv2d(g_nir1,self.df_dim*4,name='g_nir2')))
	g_bn3 = batch_norm(self.batch_size,name='g_bn3')
        g_nir3 =tf.nn.relu(g_bn3(conv2d(g_nir2,self.df_dim*2,name='g_nir3')))
	g_bn4 = batch_norm(self.batch_size,name='g_bn4')
        g_nir4 =tf.nn.relu(g_bn4(conv2d(g_nir3,self.df_dim,name='g_nir4')))
        g_nir5 =conv2d(g_nir4,3,name='g_nir5')
	return tf.tanh(g_nir5)


    def discriminator(self, image,keep_prob, reuse=False):
	if reuse:
            tf.get_variable_scope().reuse_variables()    
        h0 = lrelu(conv2d(image, self.df_dim,d_h=2,d_w=2,name='d_h0_conv')) #output size: 32x32
	d_bn1 = batch_norm(self.batch_size,name='d_bn1')
        h1 = lrelu(d_bn1(conv2d(h0, self.df_dim*2, d_h=2,d_w=2,name='d_h1_conv'))) #output size: 16 x16 
	d_bn2 = batch_norm(self.batch_size,name='d_bn2')
        h2 = lrelu(d_bn2(conv2d(h1, self.df_dim*4, d_h=2,d_w=2,name='d_h2_conv'))) #output size: 8 x 8
	d_bn3 = batch_norm(self.batch_size,name='d_bn3')
        h3 = lrelu(d_bn3(conv2d(h2, self.df_dim*8, d_h=2,d_w=2,name='d_h3_conv'))) #output size: 4x4
        h4 = conv2d(h3,1, k_h=1,k_w=1,name='d_h4_conv') #output size: 4x4
        #h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 512, 'd_h4_lin') #output 512
	#h5 = linear(h4, 1, 'd_h5_lin')
        return tf.nn.sigmoid(h4)


    def sampler(self,nir):
	tf.get_variable_scope().reuse_variables()
	g_nir0 =tf.nn.relu(conv2d(nir,self.df_dim,name='g_nir0'))
	g_bn1 = batch_norm(self.batch_size,name='g_bn1')
        g_nir1 =tf.nn.relu(g_bn1(conv2d(g_nir0,self.df_dim*8,name='g_nir1'),train=False))
	g_bn2 = batch_norm(self.batch_size,name='g_bn2')
        g_nir2 =tf.nn.relu(g_bn2(conv2d(g_nir1,self.df_dim*4,name='g_nir2'),train=False))
	g_bn3 = batch_norm(self.batch_size,name='g_bn3')
        g_nir3 =tf.nn.relu(g_bn3(conv2d(g_nir2,self.df_dim*2,name='g_nir3'),train=False))
	g_bn4 = batch_norm(self.batch_size,name='g_bn4')
        g_nir4 =tf.nn.relu(g_bn4(conv2d(g_nir3,self.df_dim,name='g_nir4'),train=False))
        g_nir5 =conv2d(g_nir4,3,name='g_nir5')
	return tf.tanh(g_nir5)

