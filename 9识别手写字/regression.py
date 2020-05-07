import os
import model
import tensorflow as tf
import input_data
data=input_data.read_data_sets('MNIST_data',one_hot=True)
#创建模型
with tf.variable_scope("regression"):
    #784=28*28
    x=tf.placeholder(tf.float32,[None,784])
    y,variables=model.regression(x)
#训练
y_=tf.placeholder("float",[None,10])
#交叉熵
cross_entropy=tf.reduce_sum(y_*tf.log(y))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction=tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
accruacy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#保存模型W，b
saver=tf.train.Saver(variables)
#开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        #每一次训练加100个数据
        batch_xs,batch_ys=data.train.next_batch(100)
        sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
        if i%100==0:
            print(sess.run(accruacy, feed_dict={x: data.test.images, y_: data.test.labels}))
    path=saver.save(
        sess,
        os.path.join(os.path.dirname(__file__),
        'data',
        'regression.ckpt'),
        write_meta_graph=False,
        write_state=False)
    print("Saved:",path)
