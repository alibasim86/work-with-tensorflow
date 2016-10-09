import numpy as np
import tensorflow as tf
import struct



def image_file_reader(file,file2):

	f = open(file,"rb")
	data = f.read(4)
	data = struct.unpack(">I",data)[0]
	data =f.read(4)
	count = struct.unpack(">I",data)[0]
	data = f.read(4)
	row = struct.unpack(">I",data)[0]
	data = f.read(4)
	col = struct.unpack(">I",data)[0]
	m = np.ndarray(shape=(count,row*col))
	pixc = 0
	j = 0
	for i in range (count*row*col):
		data = f.read(1)
		pixc = pixc+1
		pix = struct.unpack("B",data)[0]
		m [j][pixc-1] = pix
		if pixc == row*col:
			pixc = 0
			j =j + 1
	
	f = open(file2,"rb")
	data = f.read(4)
	data = f.read(4)
	count = struct.unpack(">I",data)[0]
	n = np.zeros(shape=(count,10))
	for i in range (count):
		data = f.read(1)
		v = struct.unpack("B",data)[0]
		n[i][v] = 1.0 
	m = np.multiply(m,1.0/255.0)
	return(count,m,n)



def multi_layer(i,w,b):
	layer_1 = tf.add(tf.matmul(i,w["h1"]),b["b1"])
	layer_2 = tf.add(tf.matmul(layer_1,w["h2"]),b["b2"])
	layer_2 = tf.nn.relu(layer_2)
	out = tf.add(tf.matmul(layer_2,w["out"]),b["out"])

	return(out)



c,im,la = image_file_reader("./data/train-images-idx3-ubyte","./data/train-labels-idx1-ubyte")

print c 
#print im[0]i
#print la[0]





n_input = 784
n_out = 10
l_rate = 0.5
n_hidden1 = 256
n_hidden2 = 256
t_epoches = 15
batch_size = 100
d_step = 1
v = np.zeros(shape=(batch_size,784))
w = np.zeros(shape=(batch_size,10))

we = { "h1": tf.Variable(tf.random_normal([n_input,n_hidden1])),
	"h2":tf.Variable(tf.random_normal([n_hidden1,n_hidden2])),
	"out":tf.Variable(tf.random_normal([n_hidden2,n_out]))}
ba ={ "b1": tf.Variable(tf.random_normal([n_hidden1])),
	"b2": tf.Variable(tf.random_normal([n_hidden2])),
	"out": tf.Variable(tf.random_normal([n_out]))}

x = tf.placeholder("float",[None,n_input])
y = tf.placeholder("float",[None,n_out])


pred = multi_layer(x,we,ba)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	b = 0
	for epoch in range(t_epoches):
		avg_cost = 0 
		total_batch = int(c/batch_size)
		for i in range(total_batch):
			for j in range(batch_size):
				v[j] = im[j+b]
				w[j] = la[j+b]
			_, c = sess.run([optimizer, cost], feed_dict={x:v,y:w})
			avg_cost = c / total_batch
			b = b+batch_size
		if epoch % d_step == 0:
			print "Epoch:", '%04d' % (epoch+1),"cost=","{:.9f}".format(avg_cost)
	print "Optimization Finished!"

	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) 
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

	tc,tim,tla = image_file_reader("./data/t10k-images-idx3-ubyte","./data/t10k-labels-idx1-ubyte")
	print "Accuracy:", accuracy.eval({x: tim, y: tla})
