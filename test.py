# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:10:41 2020

@author: benny
"""
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten ,Input
from keras.layers import Conv2D, MaxPooling2D, Reshape, Add
from keras.metrics import categorical_accuracy
from keras.regularizers import l1_l2, l2, l1
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from tensorflow.python.keras import backend as K
from keras.preprocessing.image import array_to_img,img_to_array
from keras.models import Sequential
from keras.models import load_model
from tensorflow.keras.layers import Embedding
import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers
from keras import backend as k
#K.set_image_dim_ordering('th')
import sys
#print(keras.__version__)
#print(tf._version_)
epsilons=20
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
dataset_D=x_train[0:200]
restOfset=x_train[200:]
model = load_model(sys.argv[2])
labels = model.predict(dataset_D)
model_B = load_model(sys.argv[3])
restOflabels=model.predict(restOfset)



'''
model_B = Sequential()


print('X_train shape:', dataset_D.shape)
print(dataset_D.shape[0], 'train samples')
#Layer 1
'''


'''
#model_B.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,32,32), data_format='channels_first'))
model_B.add(Conv2D(filters=100, kernel_size=(3,3), padding='valid', activation="relu", input_shape=(3,32,32),data_format='channels_first'))
model_B.add(MaxPooling2D(pool_size=(2,2)))

model_B.add(Conv2D(filters=100, kernel_size=(3,3), padding='valid', activation="relu"))
model_B.add(MaxPooling2D(pool_size=(2,2)))


model_B.add(Flatten())# shape equals to [batch_size, 32] 32 is the number of filters
model_B.add(Dense(2))#Fully connected layer
model_B.add(Activation('sigmoid'))
'''



'''
model_B.add(Flatten())
model_B.add(Dense(100))
model_B.add(Activation('relu'))
model_B.add(Dense(100))
model_B.add(Activation('relu'))
model_B.add(Dense(2))
model_B.add(Activation('softmax'))


#keras.utils.multi_gpu_model(model_B, gpus=2, cpu_merge=False, cpu_relocation=False)
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, decay=1e-6)

model_B.compile(loss='categorical_crossentropy',
          optimizer=opt,
         metrics=['accuracy'])

model_B.fit(dataset_D,labels,batch_size=32,epochs=10)

print('accuracy: ',model_B.evaluate(dataset_D,labels))

#sess=tf.compat.v2.InteractionSession()
#sess.run(tf.compat.v2.initialize_all_variables())
'''


def gradient(model,input_data,labels):
  #gradients = k.gradients(model.output,model.trainable_weights)
  bce = keras.losses.CategoricalCrossentropy()
  loss= bce(model.output,labels)
  gradients = k.gradients(loss,model.input)

  f = k.function([model.input],gradients)
  x=input_data
  return f([x])


epsilon=0.0625
'''
epsilon=0.0625

for i in range(10):
  for j in range(len(dataset_D)):
    temp=np.zeros(dataset_D.shape)
    #gradients=k.gradients(model_B.output,model_B.input)
    #evaluated_gradients = sess.run(gradients,feed_dict={model_B.input:dataset_D})
    evaluated_gradients = gradient(model_B,dataset_D)
    #print(evaluated_gradients)
    if np.random.randint(2)==0:
      temp[i]=dataset_D[i]+0.1*np.sign(evaluated_gradients[0][i])
    else:
      temp[i]=dataset_D[i]-0.1*np.sign(evaluated_gradients[0][i])
  labels=np.concatenate((labels,labels),0)
  dataset_D=np.concatenate((dataset_D,temp),0)
  model_B.fit(dataset_D,labels,batch_size=32,epochs=3)
  final_gradient=gradient(model_B,dataset_D)
  final_dataset=dataset_D+0.0625*np.sign(np.sum(final_gradient))
  #final_predict=model.predict(final_dataset)
  
  print(model.evaluate(final_dataset,labels))

'''





#print(dataset_D.shape)

final_gradient=gradient(model_B,restOfset,restOflabels)
final_dataset=restOfset+epsilons*np.sign(final_gradient)
#final_predict=model.predict(final_dataset)
print(model.evaluate(final_dataset[0,:,:,:,:],restOflabels))

















'''
gradients=k.gradients(model_B.output,model_B.input)
f=k.function([model.input],gradients)
x=dataset_D
print(f([x]))


gradients=keras.backend.gradients(model_B.output,model_B.input) 
trainingExample = np.random.random((1,32,32,3)) 
sess=tf.compat.v1.InteractiveSession() 
sess.run(tf.compat.v1.initialize_all_variables())
evaluated_gradients = sess.run(gradients,feed_dict={mlpmodel.input:trainingExample})
'''










'''
base_model = tf.keras.layers.Conv2D(filters=100, kernel_size=(3,3), padding='valid', activation="relu", input_shape=(3,32,32),data_format='channels_first')
base_model= tf.keras.layers.GlobalMaxPool2D()(base_model.output)
base_model = tf.keras.layers.Conv2D(filters=100, kernel_size=(3,3), padding='valid', activation="relu")(base_model)
base_model= tf.keras.layers.GlobalMaxPool2D()(base_model.output)
output = tf.keras.layers.Dense(2)(base_model)

#model_B.fit(dataset_D,labels,batch_size=4,epochs=20,shuffle=True)
#output=tf.keras.layers.Dense(2)(model_B)
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

epsilon=0.01
def loss_fun(y_true, y_pred):
    loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    return loss

x_noise=np.zeros_like(dataset_D)
x_adv=dataset_D
predictions = model_B.predict(dataset_D)
#print(prediction)

image = tf.Variable(dataset_D)
for iteration in range(20):
    with tf.GradientTape() as tape:
        tape.watch(image)
        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        prediction = model(image,training=False)  # Logits for this minibatch

        # Compute the loss value for this minibatch.
        loss_value = loss_fun(target, prediction)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, image)
    #print(grads)  # output: [None]
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer.apply_gradients(zip([grads], [image]))

    print('Iteration {}'.format(iteration))




x_noise=np.zeros_like(dataset_D)
x_adv=dataset_D
predictions = model_B.predict(dataset_D)
#print(prediction)
epsilon=0.01



gradients=keras.backend.gradients(model_B.output,model_B.input) 
trainingExample = np.random.random((1,3,32,32)) 
sess=tf.compat.v1.InteractiveSession() 
sess.run(tf.compat.v1.initialize_all_variables())
evaluated_gradients = sess.run(gradients,feed_dict={mlpmodel.input:trainingExample})
print(evaluated_gradients)




epsilon=0.01

x_noise=np.zeros_like(dataset_D)
x_adv=dataset_D
predictions = model_B.predict(dataset_D)
#print(prediction)




loss_object = tf.keras.losses.BinaryCrossentropy()

def create_adversarial_pattern(dataset_D, labels):
  image = tf.Variable(dataset_D[0])

  with tf.GradientTape() as tape:
    tape.watch(image)
    #prediction = predictions[0]
    loss = loss_object(labels[0], predictions[0])
  print(loss)
  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, tf.Variable(image))
  print(gradient)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad
perturbations = create_adversarial_pattern(dataset_D, labels)
print(perturbations)





loss = -1*K.binary_crossentropy(predictions,labels)
print(loss)

grads = K.gradients(loss,dataset_D)
print(grads)




delta = K.sign(grads)
x_noise = x_noise+delta
x_adv=x_adv+epsilon*delta

print(prediction)
print(loss)
print(grads)
print(delta)
print(x_adv)




epochs=20
epsilon=0.01
loss = 0


for i in range(epochs):
  for j in range(len(dataset_D)):
    x_noise=np.zeros_like(dataset_D[j])
    x_adv=dataset_D[j]
    prediction = model_B.predict(dataset_D[j])
    loss = -1*K.binary_crossentropy(prediction,labels[j])
    grads = K.gradients(loss,dataset_D[j])

    delta = K.sign(grads[0])    
    x_noise = x_noise+delta
    x_adv=x_adv+epsilon*delta

    



loss_object = tf.compat.v1.keras.losses.CategoricalCrossentropy

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad

labrador_retriever_index = labels[0]
label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))

perturbations = create_adversarial_pattern(dataset_D[0], label)
plt.imshow(perturbations[0])
'''



print('success')



























