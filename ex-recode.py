import time
import numpy as np
import tensorflow as tf
import math
from PIL import Image
import os

batch_size = 20
data_fir = ""
train_epoch = 500
image_width, image_height = 100,100                          
image_size = image_width * image_height
h_size = 256
sita_g0_size = 128
sita_g1_size = 128
sita_g2_size = 256
g_size = 256
glimpses = 7
reinforce_std = 0.22
class_size = 10
minRadius = 6
sensorBandWidth = minRadius * 2
sensorArea = sensorBandWidth ** 2
depth = 4
totalSensorBandwidth = depth * sensorArea
learning_rate = 1e-3
is_train = True
SHARE = None

def Glimpse_Sensor(img_list,l):
	imgs_zooms = [0] * batch_size
	for i in range(batch_size):
		imgZooms = []
		img = tf.reshape(img_list[i,:],(image_height,image_width,1))
		max_radius = minRadius * (2 ** (depth - 1))
		img = tf.image.pad_to_bounding_box(img,max_radius,max_radius,max_radius*2 + image_height,max_radius*2 + image_width)
		for j in range(depth):
			r = int(minRadius * (2 ** j))
			d_raw = 2 * r
			d = tf.constant([d_raw,d_raw])
			loc = l[i,:] + 1.0
			loc = tf.cast(loc * image_height / 2.0, tf.int32)
			img = tf.reshape(img,(img.get_shape()[0].value,img.get_shape()[1].value))
			zoom = tf.slice(img, loc - r + max_radius, d) 
			if j > 0:
				zoom = tf.reshape(zoom,[1,d_raw,d_raw,1])
				zoom = tf.nn.avg_pool(zoom,ksize=[1, 2**j, 2**j, 1],strides=[1, 2**j, 2**j, 1], padding='SAME')
			zoom = tf.reshape(zoom,[sensorArea])
			imgZooms.append(zoom)
		imgZooms = tf.concat(0,imgZooms)
		imgs_zooms[i] = tf.reshape(imgZooms,[totalSensorBandwidth])
	imgs_zooms = tf.concat(0,imgs_zooms)
	imgs_zooms = tf.reshape(imgs_zooms,(batch_size,totalSensorBandwidth))
	return imgs_zooms
def Glimpse_network(xt,loc,reu=None):
	with tf.variable_scope("Glimpse_network",reuse=reu):
		w_sita_g0 = tf.get_variable("w_sita_g0",[totalSensorBandwidth,sita_g0_size])
		b_sita_g0 = tf.get_variable("b_sita_g0",[sita_g0_size])
		sita_g0 = tf.nn.relu(tf.matmul(xt,w_sita_g0) + b_sita_g0)
		w_sita_g1 = tf.get_variable("w_sita_g1",[2,sita_g1_size])
		b_sita_g1 = tf.get_variable("b_sita_g1",[sita_g1_size])
		sita_g1 = tf.nn.relu(tf.matmul(loc,w_sita_g1) + b_sita_g1)
        
		con = tf.concat(1,[sita_g0,sita_g1])
		w_sita_g2 = tf.get_variable("w_sita_g2",[sita_g2_size,sita_g2_size])
		b_sita_g2 = tf.get_variable("b_sita_g2",[sita_g2_size])
		imagehidden = tf.nn.relu(tf.matmul(con,w_sita_g2)+b_sita_g2)
	return imagehidden
def corenetwork(h,g,reu=None):
	with tf.variable_scope("corenetwork",reuse=reu):
		w_h = tf.get_variable("w_h",[h_size,h_size])
		b_h = tf.get_variable("b_h",[h_size])
		tem = tf.matmul(h,w_h) + b_h

		w_g = tf.get_variable("w_g",[g_size,h_size])
		b_g = tf.get_variable("b_g",[h_size])
		tep = tf.matmul(g,w_g) + b_g

	return tf.nn.relu(tem + tep)

def reinforce(l,std,is_train):
	if is_train:
		loc = tf.random_normal((batch_size,2),mean=l,stddev=std)
	else:
		loc = l
	return loc
def gaussian_pdf(mean,sample):
	return tf.square(sample - mean)/ (2 * reinforce_std**2)
def get_next_l(h,reu=None):
	with tf.variable_scope("fb",reuse=reu):
		w_sita_l = tf.get_variable("w_sita_l",[h_size,2])
		b_sita_l = tf.get_variable("b_sita_l",[2])
		l = tf.tanh(tf.matmul(h,w_sita_l) + b_sita_l)
	return l
#######################get the input######################
x = tf.placeholder(tf.float32,shape=(batch_size,image_size))
label = tf.placeholder(tf.int32,shape=(batch_size,class_size))
##################init start state,and loc################
h = tf.zeros([batch_size,h_size])
l_mean = [0]*glimpses
l_sample = [0]*glimpses
l_mean[0] = tf.zeros([batch_size,2])
l_sample[0] = tf.zeros([batch_size,2])
#################      main    ########################### 
for i in range(glimpses):
	glimpse_sensors = Glimpse_Sensor(x,tf.tanh(l_sample[i]))
	g = Glimpse_network(glimpse_sensors,tf.tanh(l_sample[i]),reu=SHARE)
	h = corenetwork(h,g,reu=SHARE)
	if i < glimpses-1:
		l_mean[i+1] = get_next_l(h,reu=SHARE)
		l_sample[i+1] = reinforce(l_mean[i+1],reinforce_std,is_train)
	SHARE = True
################    classify   ###########################
with tf.variable_scope("fa"):
	w_sita_a = tf.get_variable("w_sita_a",[h_size,class_size])
	b_sita_a = tf.get_variable("b_sita_a",[class_size])
	a = tf.matmul(h,w_sita_a) + b_sita_a

################  caculate the baseline   #################
base = tf.get_variable("baseline",[1])
con = tf.ones([batch_size,1],tf.float32)
baseline = con + base
################  caculate the policy prop ################
l_mean = tf.transpose(tf.pack(l_mean), perm = [1, 0, 2])
l_sample = tf.transpose(tf.pack(l_sample), perm = [1, 0, 2])
l_prop = tf.reduce_mean(gaussian_pdf(l_mean,l_sample),2)
l_prop = tf.reduce_sum(l_prop,1)
################  caculate loss fuction    #################
softa = tf.nn.softmax(a)
p_y = tf.arg_max(softa,1)
y = tf.arg_max(label,1)
R = tf.cast(tf.equal(p_y,y),tf.float32)
acc = tf.reduce_sum(R)

tvars = tf.trainable_variables()
l2_loss = 0.0
for i in tvars:
	l2_loss += tf.nn.l2_loss(i)

classify_cost = -tf.reduce_mean(tf.reduce_sum(tf.log(softa+0.0000001) * tf.cast(label,tf.float32),1))
baseline_cost = tf.reduce_mean(tf.square(baseline - R))
reinforce_cost = tf.reduce_mean(l_prop * (R - tf.stop_gradient(baseline)))
loss = classify_cost
train_op=tf.train.MomentumOptimizer(learning_rate,momentum=0.9).minimize(reinforce_cost + classify_cost+baseline_cost+l2_loss*0.0005)
##################train and eval#####################
fetches = []
fetches.extend([acc, loss, train_op])
eval_f = []
eval_f.extend([p_y,acc, loss])

train_data = np.load("data//mnist_x.npy")
train_data = train_data.reshape(len(train_data),-1)
train_label = np.load("data//mnist_label.npy")
train_label = train_label.reshape(len(train_data),-1)
#get data and label,20000 images,use 18000train 2000test
sess=tf.InteractiveSession()
saver = tf.train.Saver() 
tf.initialize_all_variables().run()
#saver.restore(sess, "103model.ckpt")
eval_len = 100
train_len = len(train_data) // batch_size - eval_len
for i in range(train_epoch):
	train_accuary = 0
	is_train = True
	train_loss = 0
	for j in range(train_len):
		xtrain = train_data[j*batch_size:(j+1)*batch_size]
		xlabel = train_label[j*batch_size:(j+1)*batch_size]
		feed_dict = {x:xtrain,label:xlabel}
		results = sess.run(fetches,feed_dict)
		ac,lo,_ = results
		train_accuary += ac
		train_loss += lo
	learning_rate = learning_rate + (0.00001 - learning_rate) / 800	
	train_accuary = train_accuary / (train_len*batch_size)
	train_loss = train_loss / (train_len)
	is_train = False
	eval_accuary = 0
	for j in xrange(train_len,train_len+eval_len):
		xtrain = train_data[j*batch_size:(j+1)*batch_size]
		xlabel = train_label[j*batch_size:(j+1)*batch_size]
		feed_dict = {x:xtrain,label:xlabel}
		results = sess.run(eval_f,feed_dict)
		p_l,ac,co = results
		eval_accuary += ac
#		saveimage(xtrain[1].reshape(100,100),str(j)+str(tf.arg_max(xlabel)))
#	print("p_l:",p_l)
#	print("l:",xlabel)
	eval_accuary = eval_accuary / (eval_len*batch_size)
	print("epoch = %d. train_accuary = %f, eval_accuary= %f, aveage_loss_reinforce= %f" % (i,train_accuary,eval_accuary,train_loss))
	ckpt_file = os.path.join(str(i) + "model.ckpt")
	saver.save(sess,ckpt_file)
sess.close()






