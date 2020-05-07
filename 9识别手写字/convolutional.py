import os
import model
import tensorflow as tf
import input_data
data=input_data.read_data_sets('MNIST_data',one_hot=True)
#创建模型
with tf.variable_scope("convolutional"):
    #784=28*28
    x=tf.placeholder(tf.float32,[None,784])
    keep_prob=tf.placeholder(tf.float32)
    y,variables=model.convolutional(x,keep_prob)
#训练
y_=tf.placeholder(tf.float32,[None,10],name='y')
#交叉熵
cross_entropy=tf.reduce_sum(y_*tf.log(y))
train_step=tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
accruacy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#保存模型W，b
saver=tf.train.Saver(variables)
#开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        #每一次训练加100个数据
        #batch_xs,batch_ys=data.train.next_batch(50)
        batch=data.train.next_batch(50)
        # batch_xs,batch_ys=batch[0],batch[1]
        if i%10==0:
            train_accuracy=accruacy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
            print("第",i,"次，准确率为：",train_accuracy)
        sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
    print(sess.run(accruacy, feed_dict={x: data.test.images, y_: data.test.labels,keep_prob:1.0}))
    path=saver.save(
        sess,
        os.path.join(os.path.dirname(__file__),
        'data',
        'convolutional.ckpt'),
        write_meta_graph=False,
        write_state=False)
    print("Saved:",path)
