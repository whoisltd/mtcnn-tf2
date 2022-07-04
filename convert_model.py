import tensorflow as tf

model = tf.keras.models.load_model('modelss (1)/pnet.h5')
model.save('tf_model/p_net/1', save_format="tf")
print('done')
# r_net
model = tf.keras.models.load_model('modelss/rnet.h5')
model.save('tf_model/r_net/1', save_format="tf")
print('done')
# o_net
model = tf.keras.models.load_model('modelss/onet.h5')
model.save('tf_model/o_net/1', save_format="tf")
print('done')