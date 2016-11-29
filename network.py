from ops import *
class network(object):
    def __init__(self,num_block,batch_size):
	self.num_block = num_block
	self.batch_size = batch_size
    def generator(self,nir):
	g_bn0 = batch_norm(self.batch_size,name='g_bn0')
        h0 =tf.nn.relu(bn0(conv2d(nir,self.df_dim,name='g_nir0')))
	g_bn1 = batch_norm(self.batch_size,name='g_bn1')
        block =tf.nn.relu(g_bn1(conv2d(h0,20,k_h=1,k_w=1,name='g_nir1')))
	for ii in range(nu,_block):
            g_bn_block = batch_norm(self.batch_size,name='g_bn2_%s' %ii)
            block =tf.nn.relu(g_bn_block(conv2d(block,20,k_h=3,k_w=3,name='g_nir2_%s' %ii)))
        final =deconv2d(block,[self.batch_size,nir.get_shape().as_list()[1],nir.get_shape().as_list()[2],3],name='g_end',with_w=False)
	return tf.nn.tanh(final)

    def discriminator(self, image, reuse=False):
	if reuse:
            tf.get_variable_scope().reuse_variables()    
        h0 = lrelu(conv2d(image, self.df_dim, d_h=2,d_w=2,name='d_h0_conv'))
        h1 = lrelu(self.d_normal_bn1(conv2d(h0, self.df_dim*2, d_h=2,d_w=2,name='d_h1_conv')))
        h2 = lrelu(self.d_normal_bn2(conv2d(h1, self.df_dim*4, d_h=2,d_w=2,name='d_h2_conv')))
        h3 = lrelu(self.d_normal_bn3(conv2d(h2, self.df_dim*8, d_h=2,d_w=2,name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
        return tf.nn.sigmoid(h4)

