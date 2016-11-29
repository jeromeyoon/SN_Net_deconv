import os,time,pdb,argparse,threading
import time
from glob import glob
import numpy as np
from numpy import inf
import tensorflow as tf
import pdb
#from tensorflow.python.ops.script_ops import *
from ops import *
from utils import *
from random import shuffle
from network import networks
class DCGAN(object):
    def __init__(self, sess, image_size=108, is_train=True,is_crop=True,\
                 batch_size=32,num_block=1,ir_image_shape=[64, 64,1], normal_image_shape=[64, 64, 3],\
	         light_shape=[64,64,3],df_dim=64,dataset_name='default',checkpoint_dir=None):


        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.normal_image_shape = normal_image_shape
        self.ir_image_shape = ir_image_shape
        self.df_dim = df_dim
        self.dataset_name = dataset_name
	self.num_block = num_block
        self.checkpoint_dir = checkpoint_dir
	self.use_queue = True
	self.mean_nir = -0.3313 #-1~1
	self.build_model()

    def build_model(self):
	
	if not self.use_queue:

        	self.ir_images = tf.placeholder(tf.float32, [self.batch_size] + self.ir_image_shape,
                                    name='ir_images')
        	self.normal_images = tf.placeholder(tf.float32, [self.batch_size] + self.normal_image_shape,
                                    name='normal_images')
	else:
		print ' using queue loading'
		self.ir_image_single = tf.placeholder(tf.float32,shape=self.ir_image_shape)
		self.normal_image_single = tf.placeholder(tf.float32,shape=self.normal_image_shape)
		q = tf.FIFOQueue(1000,[tf.float32,tf.float32],[[self.ir_image_shape[0],self.ir_image_shape[1],1],[self.normal_image_shape[0],self.normal_image_shape[1],3]])
		self.enqueue_op = q.enqueue([self.ir_image_single,self.normal_image_single])
		self.ir_images, self.normal_images = q.dequeue_many(self.batch_size)
	"""
        self.ir_test = tf.placeholder(tf.float32, [1,600,800,1],name='ir_test')
        self.gt_test = tf.placeholder(tf.float32, [1,600,800,3],name='gt_test')
	"""
	net  = networks(self.num_block,self.batch_size,self.df_dim)
	self.G = net.generator(self.ir_images)
	self.D = net.discriminator(self.normal_images)
	self.D_  = net.discriminator(self.G,reuse=True)

	# generated surface normal
        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D), self.D)
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_), self.D_)
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.L1_loss = tf.reduce_mean(tf.abs(tf.sub(self.G,self.normal_images)))
        self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_), self.D_)
        self.gen_loss = self.g_loss + self.L1_loss

	self.saver = tf.train.Saver(max_to_keep=0)
	t_vars = tf.trainable_variables()
	self.d_vars =[var for var in t_vars if 'd_' in var.name]
	self.g_vars =[var for var in t_vars if 'g_' in var.name]
	

    def train(self, config):
        #####Train DCGAN####

        global_step = tf.Variable(0,name='global_step',trainable=False)
        global_step1 = tf.Variable(0,name='global_step1',trainable=False)
	
	d_optim = tf.train.AdamOptimizer(config.d_learning_rate,beta1=config.beta1) \
                          .minimize(self.d_loss, global_step=global_step,var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.g_learning_rate,beta1=config.beta1) \
                          .minimize(self.gen_loss, global_step=global_step1,var_list=self.g_vars)
	tf.initialize_all_variables().run()
	
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # loda training and validation dataset path
        data = json.load(open("/research2/ECCV_journal/with_light/json/traininput.json"))
        data_light = json.load(open("/research2/ECCV_journal/with_light/json/trainlight.json"))
        data_label = json.load(open("/research2/ECCV_journal/with_light/json/traingt.json"))
        datalist =[data[idx] for idx in xrange(0,len(data))]
        labellist =[data_label[idx] for idx in xrange(0,len(data))]
	shuffle(datalist)
	shuffle(labellist)

        list_val = [11,16,21,22,33,36,38,53,59,92]


	if self.use_queue:
	    # creat thread
	    coord = tf.train.Coordinator()
            num_thread =32
            for i in range(num_thread):
 	        t = threading.Thread(target=self.load_and_enqueue,args=(coord,datalist,labellist,i,num_thread))
	 	t.start()

	if self.use_queue:
	    for epoch in xrange(config.epoch):
	        #shuffle = np.random.permutation(range(len(data)))
	        batch_idxs = min(len(data), config.train_size)/config.batch_size
		sum_L1 = 0.0
		sum_g =0.0
		if epoch ==0:
		    train_log = open(os.path.join("logs",'train_%s.log' %config.dataset),'w')
		else:
	    	    train_log = open(os.path.join("logs",'train_%s.log' %config.dataset),'aw')

		for idx in xrange(0,batch_idxs):
        	     start_time = time.time()
		     _ =self.sess.run([d_optim])
		     _,g_loss,L1_loss =self.sess.run([g_optim,self.g_loss,self.L1_loss])
		     print("Epoch: [%2d] [%4d/%4d] time: %4.4f g_loss: %.6f L1_loss:%.4f" \
		     % (epoch, idx, batch_idxs,time.time() - start_time,g_loss,L1_loss))
		     sum_L1 += L1_loss 	
		     sum_g += g_loss	
		train_log.write('epoch %06d mean_g %.6f  mean_L1 %.6f\n' %(epoch,sum_g/(batch_idxs),sum_L1/(batch_idxs)))
		train_log.close()
	        self.save(config.checkpoint_dir,global_step)


	else:
	    for epoch in xrange(config.epoch):
	         # loda training and validation dataset path
	         shuffle_ = np.random.permutation(range(len(data)))
	         batch_idxs = min(len(data), config.train_size)/config.batch_size
		    
	         for idx in xrange(0, batch_idxs):
        	     start_time = time.time()
		     batch_files = shuffle_[idx*config.batch_size:(idx+1)*config.batch_size]
    		     batches = [get_image(datalist[batch_file],labellist[batch_file],self.image_size,np.random.randint(64,224-64),\
					np.random.randint(64,224-64), is_crop=self.is_crop) for batch_file in batch_files]

		     batches = np.array(batches).astype(np.float32)
		     batch_images = np.reshape(batches[:,:,:,0],[config.batch_size,64,64,1])
		     batchlabel_images = np.reshape(batches[:,:,:,1:],[config.batch_size,64,64,3])
		     #mask_mean = batch_mask * self.mean_nir
		     #batch_images = batch_images- mask_mean
		     # Update Normal D network
		     _= self.sess.run([d_optim], feed_dict={self.ir_images: batch_images,self.normal_images:batchlabel_images })
		     self.writer.add_summary(summary_str, global_step.eval())

		     # Update NIR G network
		     _,g_loss,L1_loss = self.sess.run([g_optim,self.g_loss,self.L1_loss], feed_dict={ self.ir_images: batch_images,self.normal_images:batchlabel_images})
		     print("Epoch: [%2d] [%4d/%4d] time: %4.4f g_loss: %.6f L1_loss:%.4f" \
		     % (epoch, idx, batch_idxs,time.time() - start_time,g_loss,L1_loss))
	         self.save(config.checkpoint_dir,global_step)
    
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name,self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

	    
    def load_and_enqueue(self,coord,file_list,label_list,idx=0,num_thread=1):
	count =0;
	length = len(file_list)
	while not coord.should_stop():
	    i = (count*num_thread + idx) % length;
            input_img = scipy.misc.imread(file_list[i]).reshape([224,224,1]).astype(np.float32)
	    gt_img = scipy.misc.imread(label_list[i]).reshape([224,224,3]).astype(np.float32)
	    input_img = input_img/127.5 -1.
	    gt_img = gt_img/127.5 -1.
	    rand_x = np.random.randint(64,224-64)
	    rand_y = np.random.randint(64,224-64)
            self.sess.run(self.enqueue_op,feed_dict={self.ir_image_single:input_img[rand_y:rand_y+64,rand_x:rand_x+64],self.normal_image_single:gt_img[rand_y:rand_y+64,rand_x:rand_x+64]})
	    count +=1
		
