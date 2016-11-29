from ops import *
import tensorflow as tf

class networks(object):
    def __init__(self,num_block,batch_size,df_dim):
	self.num_block = num_block
	self.batch_size = batch_size
	self.df_dim = df_dim
    def generator(self,nir):
	g_bn0 = batch_norm(self.batch_size,name='g_bn0')
        h0 =tf.nn.relu(g_bn0(conv2d(nir,self.df_dim,name='g_nir0')))
	g_bn1 = batch_norm(self.batch_size,name='g_bn1')
        block =tf.nn.relu(g_bn1(conv2d(h0,20,k_h=1,k_w=1,name='g_nir1')))
	for ii in range(self.num_block):
            g_bn_block = batch_norm(self.batch_size,name='g_bn2_%s' %ii)
            block =tf.nn.relu(g_bn_block(conv2d(block,20,k_h=3,k_w=3,name='g_nir2_%s' %ii)))
        final =deconv2d(block,[self.batch_size,nir.get_shape().as_list()[1],nir.get_shape().as_list()[2],3],name='g_end',with_w=False)
	return tf.nn.tanh(final)

    def discriminator(self, image, reuse=False):
	if reuse:
            tf.get_variable_scope().reuse_variables()    
        h0 = lrelu(conv2d(image, self.df_dim, d_h=2,d_w=2,name='d_h0_conv'))
	d_bn1 = batch_norm(self.batch_size,name='d_bn1')
        h1 = lrelu(d_bn1(conv2d(h0, self.df_dim*2, d_h=2,d_w=2,name='d_h1_conv')))
	d_bn2 = batch_norm(self.batch_size,name='d_bn2')
        h2 = lrelu(d_bn2(conv2d(h1, self.df_dim*4, d_h=2,d_w=2,name='d_h2_conv')))
	d_bn3 = batch_norm(self.batch_size,name='d_bn3')
        h3 = lrelu(d_bn3(conv2d(h2, self.df_dim*8, d_h=2,d_w=2,name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
        return tf.nn.sigmoid(h4)
    def sampler(self,nir):
	tf.get_variable_scope().reuse_variables()
	g_bn0 = batch_norm(self.batch_size,name='g_bn0')
        h0 =tf.nn.relu(g_bn0(conv2d(nir,self.df_dim,name='g_nir0')))
	g_bn1 = batch_norm(self.batch_size,name='g_bn1')
        block =tf.nn.relu(g_bn1(conv2d(h0,20,k_h=1,k_w=1,name='g_nir1')))
	for ii in range(self.num_block):
            g_bn_block = batch_norm(self.batch_size,name='g_bn2_%s' %ii)
            block =tf.nn.relu(g_bn_block(conv2d(block,20,k_h=3,k_w=3,name='g_nir2_%s' %ii)))
        final =deconv2d(block,[self.batch_size,nir.get_shape().as_list()[1],nir.get_shape().as_list()[2],3],name='g_end',with_w=False)
	return tf.nn.tanh(final)

