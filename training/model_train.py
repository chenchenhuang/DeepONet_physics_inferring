import numpy as np
import tensorflow as tf
import scipy
import scipy.io
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
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

def train(train_x1,train_x2,val_x1,val_x2,train_y,val_y):

  tf.keras.backend.clear_session()
  input_dim_1=2
  input_dim_2=2
  dot_dim=30
  width=15 #15
  depth=15 #10
  reg_param=1e-6
  learning_rate=1e-3
  model=MyModel(input_dim_1=input_dim_1,input_dim_2=input_dim_2,dot_dim=dot_dim,width=width,depth=depth,reg_param=reg_param)
  
  optimizer   = keras.optimizers.Adam(learning_rate=learning_rate) 
  max_epoch   = 1000000
  batch_size  = 128
  # loss_int=[]
  # loss_bc=[]
  los=[]
  val_los = []
  for epoch in range(1,max_epoch+1):
      # print(epoch)
      train_batch_index=np.random.choice(len(train_y),batch_size)
      val_batch_index=np.random.choice(len(val_y),batch_size)
      with tf.GradientTape() as tape1:
        out=model(train_x1[train_batch_index,:],train_x2[train_batch_index,:])
        val_out = model(val_x1[val_batch_index,:],val_x2[val_batch_index,:])
        # model.summary()
        # print(tf.shape(out))
        loss=tf.reduce_mean(tf.square(out-train_y[train_batch_index]))
        val_loss = tf.reduce_mean(tf.square(val_out-val_y[val_batch_index]))
      los.append(loss.numpy())
      val_los.append(val_loss.numpy())

      grads = tape1.gradient(loss, model.trainable_variables)
      
      optimizer.apply_gradients(zip(grads, model.trainable_variables)) # zip used to create an iterator over the tuples
      if epoch % 100 == 0 or epoch==max_epoch:
        print(f"Epoch: {epoch}, loss: {loss:.2e}, val_loss: {val_loss:.2e}")
        
      if epoch % 5000 == 0  or epoch==max_epoch:
        weight_path = path + 'trained_model/model_15_15_30/model_{}'.format(epoch)
        model.save_weights(weight_path)


  return model,los,val_los #,loss_int,loss_bc

class DataPrep():
  def __init__(self,path):
      self.path = path
  
  def loading(self):
    with h5py.File(path + 'train_dataset_two.mat', 'r') as file:
      train_data = list(file['data_train'])

    with h5py.File(path + 'val_dataset_two.mat', 'r') as file:
      val_data = list(file['data_val'])

    train_y = train_data[4]

    train_x1 = np.zeros((len(train_y),2))
    train_x1[:,0] = train_data[2]
    train_x1[:,1] = train_data[3]

    train_x2 = np.zeros((len(train_x1),2))
    train_x2[:,0] = train_data[0]
    train_x2[:,1] = train_data[1]

    val_y = val_data[4]
    val_x1 = np.zeros((len(val_y),2))
    val_x1[:,0] = val_data[2]
    val_x1[:,1] = val_data[3]
    val_x2 = np.zeros((len(val_x1),2))
    val_x2[:,0] = val_data[0]
    val_x2[:,1] = val_data[1]
    
    return train_x1,train_x2,val_x1,val_x2,train_y,val_y

if __name__ == '__main__':
  
  path = './'
  
  data_load = DataPrep(path)
  
  train_x1,train_x2,val_x1,val_x2,train_y,val_y = data_load.loading()
  
  model,train_loss,validation_loss=train(train_x1,train_x2,val_x1,val_x2,train_y,val_y)

  train_loss = train_loss
  val_loss = validation_loss

  scipy.io.savemat(path+'/trained_info/loss.mat', {'train_loss': train_loss,'val_loss':val_loss})


  # predict_a = np.zeros((1000*1000,2));
  # predict_a[:,0] = predict_a[:,0]+0.0286;
  # predict_a[:,1] = predict_a[:,1]+2.9286;

  # with h5py.File('/Users/huangchenchen/Dropbox/ame508_testing/grid.mat', 'r') as file:
  #     x = list(file['x'])
  #     t = list(file['t'])

  # x_t = np.zeros((1000*1000,2))
  # x = np.reshape(x,(1000*1000,))
  # t = np.reshape(t,(1000*1000,))
  # x_t[:,0] = x;
  # x_t[:,1] = t;

  # output = model.call(predict_a,x_t)

  # predict_output = output.numpy()
  # scipy.io.savemat('/Users/huangchenchen/Dropbox/ame508_testing/output_test.mat', {'output': predict_output})