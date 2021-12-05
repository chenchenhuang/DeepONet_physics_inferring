# partial parallel colormap computing

# mapping_model_loading
import numpy as np
import tensorflow as tf
import scipy
import scipy.io
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from keras.regularizers import l2
import h5py
import multiprocessing
from tqdm import tqdm

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

def dataloading(file_name,grid_file_name,full_file_name,full_grid_file_name):

    file=scipy.io.loadmat(file_name)
    known_data = file['sample_u']

    file = scipy.io.loadmat(full_file_name)
    full_known_data = file['u']
    # with h5py.File(file_name, 'r') as file:
    #   known_data = list(file['u'])
    #   known_data = np.array(known_data)
    file = scipy.io.loadmat(grid_file_name)
    x = file['x']
    t = file['t']

    x_t = np.zeros((len(x),2))
    x = np.reshape(x,(len(x),))
    t = np.reshape(t,(len(x),))
    x_t[:,0] = x;
    x_t[:,1] = t;
    #####
    file = scipy.io.loadmat(full_grid_file_name)
    x = file['x']
    t = file['t']

    full_x_t = np.zeros((len(x),2))
    x = np.reshape(x,(len(x),))
    t = np.reshape(t,(len(x),))
    full_x_t[:,0] = x;
    full_x_t[:,1] = t;

    return x_t, known_data,full_x_t,full_known_data


def modeloading(model_file_name):
  input_dim_1=2
  input_dim_2=2
  dot_dim=30
  width=15
  depth=15
  reg_param=1e-6
  learning_rate=1e-3

  model=MyModel(input_dim_1=input_dim_1,input_dim_2=input_dim_2,dot_dim=dot_dim,width=width,depth=depth,reg_param=reg_param)

#   file_name = '/Users/huangchenchen/Dropbox/AME508_Project/double_para/trained_model/model_cc/model_1000000'
  model.load_weights(model_file_name)
  
  return model

def parafinder(model,x_t,known_data,full_x_t,full_known_data):
  
    a = tf.Variable([[5,5]], name='a', trainable=True, dtype=tf.float32)
    # a1 = tf.Variable(0.05, dtype=tf.float32, name='a1')
    # a2 = tf.Variable(5.0, dtype=tf.float32, name='a2')

    optimizer   = keras.optimizers.Adam(1e-1) 
    max_epoch = 500
    k1 = 9/(1e-1-1e-4)
    k2 = 1-1e-4*k1
    los=[]
    approx_a = []
    # full_los = []
    for epoch in tqdm(range(1,max_epoch+1),dynamic_ncols=True):
      
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

        # full_output = model.call(c,full_x_t)
        # full_loss = tf.reduce_mean(tf.square(full_output-full_known_data))
        # full_los.append(full_loss.numpy())
        los.append(loss.numpy())

        approx_a.append(output_a)
        if epoch % 10 == 0 or epoch==max_epoch:
            print(f"Epoch: {epoch}, a: {output_a}, loss: {loss:.2e}")
    
    b = tf.subtract(a,tf.Variable([k2,0]))
    c = tf.multiply(b,tf.Variable([1/k1,1]))
    full_output = model.call(c,full_x_t)
    full_loss = tf.reduce_mean(tf.square(full_output-full_known_data))
    full_los = full_loss.numpy()

    return approx_a,los,full_los

def compute(n):
    load_full_file_name = '/home/chenchen/ame508/observation_data/' + str(n) + '.mat'
    full_grid_file_name = '/home/chenchen/ame508/double_para/grid_v7.mat'

    load_file_name = '/home/chenchen/ame508/partial_observation_data/' + str(n) + '.mat'
    save_file_name = '/home/chenchen/ame508/partial_parafind/opt' + str(n) + '.mat'
    model_file_name = '/home/chenchen/ame508/trained_model/model_1000000'
    grid_file_name = '/home/chenchen/ame508/double_para/sample_grid.mat'
    x_t,known_data,full_x_t,full_known_data = dataloading(load_file_name,grid_file_name,load_full_file_name,full_grid_file_name)
    model = modeloading(model_file_name)
    a,loss,full_loss = parafinder(model,x_t,known_data,full_x_t,full_known_data)
    scipy.io.savemat(save_file_name, {'a': a,'loss':loss,'full_loss':full_loss})
    

if __name__ == '__main__':
  
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool()
    #   # cnt = 0
    #   # for _ in pool.imap_unordered(Computation, range(20)):
    #   #     sys.stdout.write('done %d/%d\r' % (cnt, len(range(20))))
    #   #     cnt += 1

    for _ in tqdm(pool.imap_unordered(compute, range(1,101)), total=len(range(1,101)), dynamic_ncols=True):
        pass
    