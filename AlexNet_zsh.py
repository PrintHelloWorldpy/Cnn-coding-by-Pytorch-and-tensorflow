import tensorflow as tf
import numpy as np


class Alexnet_zsh:
    """
    自定义Alexnet类
    """
    image_size = 227
    
    def __init__(self, images = None, img_labels = None, step = 100, batch_size = 32, keep_prob = 0.5, class_num = 10,
                 start_learning_rate = 0.9, decay_rate = 0.9, decay_steps = 10, penalize_rate = 0.9):
    
        self.step = step #迭代次数
        self.batch_size = batch_size #batch大小
        self.keep_prob = keep_prob #dropout概率
        self.class_num = class_num #类别数
        self.start_learning_rate = start_learning_rate #初始学习率
        self.decay_rate = decay_rate #指数衰减率
        self.decay_steps = decay_steps #指数衰减步长
        self.penalize_rate = penalize_rate #惩罚项
        self.global_step = 0 #训练轮次
        self.images = images #训练图片数据
        self.img_labels = img_labels #训练标签数据
        if self.images == None: #如果没输入数据，随机生成数据
            self.images = tf.Session().run(tf.random_normal([10 * self.batch_size, self.image_size, self.image_size, 3],
                                 dtype = tf.float32,stddev = 0.1 ))
            one_hot_labels = np.zeros((10 * self.batch_size, self.class_num))
            for i in range(len(one_hot_labels)):
                one_hot_labels[i][np.random.randint(0, class_num)] = 1
            self.img_labels = tf.Session().run(tf.constant(one_hot_labels, dtype = tf.float32))
        self.dataset_size = self.images.shape[0] #获取长度
        self.batch_num = self.dataset_size // self.batch_size + (self.dataset_size % self.batch_size != 0)

    
    def Alexnet_network(self, batch_images, batch_img_labels):
        """
        Alexnet loss_function
        """
        #第一个卷积层,包含池化层
        with tf.name_scope('conv1') as scope:
            """
                         images: 227*227*3  
            batch_size * kernel1: 11*11*3*96 stride:4*4 padding:vaild
                         pool1: 3*3 stride:2*2
            
                         input: images 227*227*3 
            batch_size * middle: conv1 55*55*96
                         output: pool1 27*27*96
            """
            kernel1 = tf.Variable(tf.truncated_normal([11, 11, 3, 96], 
                                 dtype = tf.float32, mean = 0, stddev = 0.1), name = 'kernel1') #卷积核
            conv1 = tf.nn.conv2d(batch_images, kernel1, [1,4,4,1], padding = 'VALID', name = 'conv1') #卷积层
                               #输入    卷积核    1,步长,步长,1     SAME和VALID
            biases1 = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),  #一个值 + shape == 列表
                       trainable = True, name = "biases1")#卷积层偏移量
                    #是否可训练，默认True
            perception1 = tf.nn.bias_add(conv1, biases1) # 感知值 w*x+b    tf.nn.bias_add()可广播  tf.add()矩阵+矩阵/数
            activation1 = tf.nn.relu(perception1, name = scope) #relu层 命名为conv1
            
            lrn1 = tf.nn.lrn(conv1, depth_radius = 4, bias = 1, alpha = 0.001 / 9, beta = 0.75, name = "lrn1") #lrn层，减慢速度, 效果不好
                                 #涉及到的周围元素  修正(防止=0)  alpha参数        beta参数
            pool1 = tf.nn.max_pool(lrn1, ksize=[1,3,3,1], strides=[1,2,2,1], padding="VALID", name="pool1")
                                        #1,感受野,感受野,1    1,步长,步长，1  
           
            
        #第二个卷积层，包含池化层
        with tf.name_scope('conv2') as scope:
            """
                         pool1: 27*27*96
            batch_size * kernel2: 5*5*96*256 stride:1*1 padding:same
                         pool2: 3*3  stride:2*2 
            
                         input: pool1 27*27*96]
            batch_size * middle: conv2 27*27*256
                         output: pool2 13*13*256
            """
            kernel2 = tf.Variable(tf.truncated_normal([5, 5, 96, 256], 
                                 dtype = tf.float32, mean = 0, stddev = 0.1), name = 'kernel2') #卷积核
            conv2 = tf.nn.conv2d(pool1, kernel2, [1,1,1,1], padding = 'SAME', name = 'conv2') #卷积层
                               #输入    卷积核    1,步长,步长,1     SAME和VALID
            biases2 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),  #一个值 + shape == 列表
                       trainable = True, name = "biases2")#卷积层偏移量
                    #是否可训练，默认True
            perception2 = tf.nn.bias_add(conv2, biases2) # 感知值 w*x+b    tf.nn.bias_add()可广播  tf.add()矩阵+矩阵/数
            activation2 = tf.nn.relu(perception2, name = scope) #relu层 命名为conv2
            
            lrn2 = tf.nn.lrn(conv2, depth_radius = 4, bias = 1, alpha = 0.001 / 9, beta = 0.75, name = "lrn2") #lrn层，减慢速度, 效果不好
                                 #涉及到的周围元素  修正(防止=0)  alpha参数        beta参数
            pool2 = tf.nn.max_pool(lrn2, ksize=[1,3,3,1], strides=[1,2,2,1], padding="VALID", name="pool2")
                                        #1,感受野,感受野,1     1,步长,步长，1  
        
        
        #第三个卷积层，无池化层
        with tf.name_scope('conv3') as scope:
            """
            batch_size * pool2: 13*13*256
                         kernel3: 3*3*256*384 stride:1*1 padding:same
            
            batch_size * input: pool2 13*13*256
                         output: conv3 13*13*384
            """
            kernel3 = tf.Variable(tf.truncated_normal([3, 3, 256, 384], 
                                 dtype = tf.float32, mean = 0, stddev = 0.1), name = 'kernel3') #卷积核
            conv3 = tf.nn.conv2d(pool2, kernel3, [1,1,1,1], padding = 'SAME', name = 'conv3') #卷积层
                               #输入    卷积核    1,步长,步长,1     SAME和VALID
            biases3 = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),  #一个值 + shape == 列表
                       trainable = True, name = "biases3")#卷积层偏移量
                    #是否可训练，默认True
            perception3 = tf.nn.bias_add(conv3, biases3) # 感知值 w*x+b    tf.nn.bias_add()可广播  tf.add()矩阵+矩阵/数
            activation3 = tf.nn.relu(perception3, name = scope) #relu层 命名为conv3
          
            
        #第四个卷积层，无池化层
        with tf.name_scope('conv4') as scope:
            """
            batch_size * activation3: 13*13*256
                         kernel4: 3*3*384*384 stride:1*1 padding:same
            
            batch_size * input: activation3 13*13*384
                         output: conv4 13*13*384
            """
            kernel4 = tf.Variable(tf.truncated_normal([3, 3, 384, 384], 
                                 dtype = tf.float32, mean = 0, stddev = 0.1), name = 'kernel4') #卷积核
            conv4 = tf.nn.conv2d(activation3, kernel4, [1,1,1,1], padding = 'SAME', name = 'conv4') #卷积层
                               #输入    卷积核    1,步长,步长,1     SAME和VALID
            biases4 = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),  #一个值 + shape == 列表
                       trainable = True, name = "biases4")#卷积层偏移量
                    #是否可训练，默认True
            perception4 = tf.nn.bias_add(conv4, biases4) # 感知值 w*x+b    tf.nn.bias_add()可广播  tf.add()矩阵+矩阵/数
            activation4 = tf.nn.relu(perception4, name = scope) #relu层 命名为conv3    
            
                                
        #第五个卷积层，包含池化层
        with tf.name_scope('conv5') as scope:
            """
                         activation4: 13*13*384
            batch_size * kernel5: 3*3*384*256 stride:1*1 padding:same
                         pool5: 3*3 stride:2*2 
            
                         input:  activation4 13*13*384
            batch_size * middle: conv5 13*13*256
                         output: pool5 6*6*256
            """
            kernel5 = tf.Variable(tf.truncated_normal([3, 3, 384, 256], 
                                 dtype = tf.float32, mean = 0, stddev = 0.1), name = 'kernel5') #卷积核
            conv5 = tf.nn.conv2d(activation4, kernel5, [1,1,1,1], padding = 'SAME', name = 'conv5') #卷积层
                               #输入    卷积核    1,步长,步长,1     SAME和VALID
            biases5 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),  #一个值 + shape == 列表
                       trainable = True, name = "biases5") #卷积层偏移量
                    #是否可训练，默认True
            perception5 = tf.nn.bias_add(conv5, biases5) # 感知值 w*x+b    tf.nn.bias_add()可广播  tf.add()矩阵+矩阵/数
            activation5 = tf.nn.relu(perception5, name = scope) #relu层 命名为conv2
            
            pool5 = tf.nn.max_pool(activation5, ksize=[1,3,3,1], strides=[1,2,2,1], padding="VALID", name="pool5")
                                        #1,感受野,感受野,1     1,步长,步长，1  
    
        #第六个全连接层
        with tf.name_scope('fc6') as scope:
            """
            batch_size * pool5: 6*6*256
                         weights6: (6*6*256) * 4096 
            
            batch_size * input: pool5 6*6*256
                         output: fc6 4096
            """
            
            weights6 =  tf.Variable(tf.truncated_normal([(6*6*256), 4096], 
                                 dtype = tf.float32, mean = 0, stddev = 0.1), name = 'weights6') #全连接层权重
            biases6 = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                        trainable = True, name = "biases6")#全连接层偏移量
            flat6 = tf.reshape(pool5, [-1, 6*6*256] )  # 整形成 m*n, m = batch_size, n = 6*6*256
            perception6 = tf.add(tf.matmul(flat6, weights6), biases6)
            activation6 = tf.nn.relu(perception6, name = scope)
            
            fc6 = tf.nn.dropout(activation6, self.keep_prob) #dropout
        
        #第七个全连接层
        with tf.name_scope('fc7') as scope:
            """
            batch_size * fc6: 4096
                         weights7: 4096*4096
            
            batch_size * input: fc6 4096
                         output: fc7 4096
            """
            weights7 =  tf.Variable(tf.truncated_normal([4096, 4096], 
                                 dtype = tf.float32, mean = 0, stddev = 0.1), name = 'weights6') #全连接层权重
            biases7 = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                        trainable = True, name = "biases7")#全连接层偏移量
            perception7 = tf.add(tf.matmul(fc6, weights7), biases6)
            activation7 = tf.nn.relu(perception7, name = scope)
            
            fc7 = tf.nn.dropout(activation7, self.keep_prob) #dropout   
        
        #第八个全连接层
        with tf.name_scope('fc8') as scope:
            """
            batch_size * fc7: 4096
                         weights8: 4096*1000
            
            batch_size * input: fc7 4096
                         output: fc8 1000
            """
            weights8 =  tf.Variable(tf.truncated_normal([4096, 1000], 
                                 dtype = tf.float32, mean = 0, stddev = 0.1), name = 'weights6') #全连接层权重
            biases8 = tf.Variable(tf.constant(0.0, shape=[1000], dtype=tf.float32),
                        trainable = True, name = "biases8")#全连接层偏移量
            perception8 = tf.add(tf.matmul(fc7, weights8), biases8)
            activation8 = tf.nn.relu(perception8, name = scope)
            
            fc8 = tf.nn.dropout(activation8, self.keep_prob) #dropout   
       
        #第九个softmax层
        with tf.name_scope('softmax9') as scope:
            """
            batch_size * fc8: 1000
                         weights8: 1000*k
            
            batch_size * input: fc8 1000
                         output: softmax9 k
            """
            weights9 =  tf.Variable(tf.truncated_normal([1000, self.class_num], 
                                 dtype = tf.float32, mean = 0, stddev = 0.1), name = 'weights6') #全连接层权重
            biases9 = tf.Variable(tf.constant(0.0, shape=[self.class_num], dtype=tf.float32),
                        trainable = True, name = "biases9")#全连接层偏移量
            perception9 = tf.add(tf.matmul(fc8, weights9), biases9)
            activation9 = tf.nn.softmax(perception9, name = scope)
        
        return activation9


    def train_Alexnet_zsh(self):
        loss_functions = []
        tf.reset_default_graph()
        with tf.Session() as sess:
            batch_images = tf.placeholder(tf.float32, name = 'batch_images')
            batch_img_labels = tf.placeholder(tf.float32, name = 'batch_img_labels')
            networt_output = self.Alexnet_network(batch_images, batch_img_labels)
            init_op = tf.initialize_all_variables()
            sess.run(init_op) #初始化参数
            for i in range(self.step): #迭代
                loss_functions.append([])
                for j in range(self.batch_num): #遍历batch
                    start = (j * self.batch_size) % self.dataset_size
                    end = min(start + self.batch_size, self.dataset_size)
                    images_batch = self.images[start: end] 
                    img_labels_batch = self.img_labels[start: end] 
                    self.learning_rate = tf.train.exponential_decay(learning_rate = self.start_learning_rate,
                                        global_step = self.global_step,
                                        decay_steps = self.decay_steps, 
                                        decay_rate = self.decay_rate) #学习率    
                    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = batch_img_labels, logits = networt_output))
                    #loss_function = -tf.reduce_mean(self.img_labels * tf.log(tf.clip_by_value(activation9, 1e-10, 1.0)), axis = 1) 
                                        #求-均值           y实际     *  log(y估计)(tf.clip_by_value限定范围，防止过大过小)  #平均值坐标轴  
                    test_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss_function)  #训练一步结果
                    sess.run(test_step, feed_dict = {batch_images: images_batch, batch_img_labels: img_labels_batch}) #进行一步训练
                    loss_functions[-1].append( sess.run(loss_function, feed_dict = {batch_images: images_batch, batch_img_labels: img_labels_batch}))
                self.global_step += 1 #轮次记录
                print(loss_functions[-1])
            return sess


#文件入口
if __name__ == "__main__":        
    Alexnet_test = Alexnet_zsh()
    ans = Alexnet_test.train_Alexnet_zsh()
    
        