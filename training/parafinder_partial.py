# parafinder_partial

import numpy as np
import tensorflow as tf
import scipy
import scipy.io
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from keras.regularizers import l2
import h5py

class MyModel(tf.keras.Model):

  def __init__(self,input_dim_1,input_dim_2,dot_dim,width,depth,reg_param=1e-6):
    super(MyModel, self).__init__()
    Reg_Func = l2
    self.depth=depth
    self.layers1=[]
    self.layers2=[]

    for i  in range(self.depth-1):
      bias_ini=keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None)
      kernel_ini = keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None)
      self.layers1.append(tf.keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer=kernel_ini, bias_initializer=bias_ini,kernel_regularizer=Reg_Func(reg_param)))
      self.layers2.append(tf.keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer=kernel_ini, bias_initializer=bias_ini,kernel_regularizer=Reg_Func(reg_param)))
    self.layers1.append(tf.keras.layers.Dense(dot_dim, activation=tf.nn.tanh, kernel_initializer=kernel_ini, bias_initializer=bias_ini,kernel_regularizer=Reg_Func(reg_param)))
    self.layers2.append(tf.keras.layers.Dense(dot_dim, activation=tf.nn.tanh, kernel_initializer=kernel_ini, bias_initializer=bias_ini,kernel_regularizer=Reg_Func(reg_param)))
  
  def call(self, inputs_1, inputs_2):
    for i in range(self.depth):
      layer1=self.layers1[i]
      layer2=self.layers2[i]
      inputs_1=layer1(inputs_1)
      inputs_2=layer2(inputs_2)
    return tf.reduce_sum( tf.multiply( inputs_1, inputs_2 ), 1 )

def dataloading(data_file_name):

    file=scipy.io.loadmat(data_file_name)
    known_data = file['sample_u']

    file = scipy.io.loadmat('/Users/huangchenchen/Dropbox/AME508_Project/data_generating/test_data/sample_grid.mat')
    x = file['x']
    t = file['t']
    # with h5py.File(file_name, 'r') as file:
    #   known_data = list(file['u'])
    #   known_data = np.array(known_data)
    #   with h5py.File('/Users/huangchenchen/Dropbox/AME508_Project/data_generating/test_data/sample_grid.mat', 'r') as file:
    #     x = list(file['x'])
    #     t = list(file['t'])
    
    x_t = np.zeros((len(x),2))
    x = np.reshape(x,(len(x),))
    t = np.reshape(t,(len(x),))
    x_t[:,0] = x;
    x_t[:,1] = t;
  
    return x_t, known_data

def modeloading():
  input_dim_1=2
  input_dim_2=2
  dot_dim=30
  width=15
  depth=15
  reg_param=1e-6
  learning_rate=1e-3

  model=MyModel(input_dim_1=input_dim_1,input_dim_2=input_dim_2,dot_dim=dot_dim,width=width,depth=depth,reg_param=reg_param)

  file_name = '/Users/huangchenchen/Dropbox/AME508_Project/double_para/trained_model/model_cc/model_1000000'
  model.load_weights(file_name)
  
  return model

def parafinder(model,x_t,known_data):
  
  a = tf.Variable([[5,5]], name='a', trainable=True, dtype=tf.float32)
  # a1 = tf.Variable(0.05, dtype=tf.float32, name='a1')
  # a2 = tf.Variable(5.0, dtype=tf.float32, name='a2')

  optimizer   = keras.optimizers.Adam(1e-1) 
  max_epoch = 500
  k1 = 9/(1e-1-1e-4)
  k2 = 1-1e-4*k1

  for epoch in range(1,max_epoch+1):
      
      with tf.GradientTape() as tape:
          b = tf.subtract(a,tf.Variable([k2,0]))
          c = tf.multiply(b,tf.Variable([1/k1,1]))
          predict_output = model.call(c,x_t)
          # predict_output = output.numpy()

          loss = tf.reduce_mean(tf.square(predict_output-known_data))

      trainable_variables = [a]
      grads = tape.gradient(loss, trainable_variables)
      output_a = a.numpy()
      output_a[0,0] = (output_a[0,0]-k2)/k1
      optimizer.apply_gradients(zip(grads, trainable_variables))
      if epoch % 10 == 0 or epoch==max_epoch:
          print(f"Epoch: {epoch}, a: {output_a}, loss: {loss:.2e}")
  
  return a

if __name__ == '__main__':
  
  data_file_name = '/Users/huangchenchen/Dropbox/AME508_Project/data_generating/test_data/sample_u.mat'
  x_t,known_data = dataloading(data_file_name)
  model = modeloading()
  a = parafinder(model,x_t,known_data)
  
  # cores = multiprocessing.cpu_count()
  #   pool = multiprocessing.Pool(None,limit_cpu)
  #   # cnt = 0
  #   # for _ in pool.imap_unordered(Computation, range(20)):
  #   #     sys.stdout.write('done %d/%d\r' % (cnt, len(range(20))))
  #   #     cnt += 1

  #   for _ in tqdm(pool.imap_unordered(Computation, range(20)), total=len(range(20)), dynamic_ncols=True):
  #       pass