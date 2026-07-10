# %%
# Import libraries
from tensorflow import keras
import tensorflow as tf
# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import random 
from sklearn.preprocessing import StandardScaler
import scipy
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import entropy

# GPU setup
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
      tf.config.experimental.set_memory_growth(physical_devices[k], True)
      print('memory growth:', tf.config.experimental.get_memory_growth(
          physical_devices[k]))

batch_size = 50
data_size = 20000
wave_length = 10000

modelname = '13'
# modelname = 'nm13' # uncomment to use the model without MINE (L2-only)


# %%
# data1 = np.load('data1_10000_m2d2s.npy')
# data2 = np.load('data2_10000_m2d2s.npy')

data1 = np.load('Chaos_Signals/data1_103.npy')
data2 = np.load('Chaos_Signals/data2_103.npy')

X = data1[:,:wave_length,:] + data2[:,:wave_length,:]
Y = np.concatenate([data1[:,:wave_length,:], data2[:,:wave_length,:]], 2)
# Split data
x_train = X
x_val = X
y_train = Y
y_val = Y

def random_batch(X, y, batch_size):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]

# %%
# Model
class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.gru = GRU(100, return_sequences=True, input_shape=(None,3))
        self.dense1 = Dense(6)

    def call(self, inputs):
        self.x = self.gru(inputs)
        return self.dense1(self.x)

    def hidden_layer(self, input):
        return self.gru(input)

mymodel2 = MyModel()
mymodel2.load_weights("model_" + modelname)

# %%
Predictdata_t = mymodel2.predict(x_train[:50])
x_val = x_train[:1000]
y_val = y_train[:1000]
    
# Plot example predictions
fig, ax = plt.subplots(3, 2, figsize=(10, 7))

# Reduce the vertical spacing between subplots
plt.subplots_adjust(hspace=0.3)

titles = ["Lorenz system", "Rössler system"]

for j in range(3):
    # Left column: Lorenz output
    ax[j][0].plot(range(0, len(Predictdata_t[0])-9000), Predictdata_t[0,500:1500,j], color="b", label="Predict")
    ax[j][0].plot(range(0, len(Predictdata_t[0])-9000), y_val[0,500:1500,j], color="r", linestyle='--', label="Target")
    ax[j][0].legend()

    if j == 0:
        ax[j][0].set_title(titles[0])

    # Right column: Rossler output
    ax[j][1].plot(range(0, len(Predictdata_t[0])-9000), Predictdata_t[0,500:1500,j+3], color="b", label="Predict")
    ax[j][1].plot(range(0, len(Predictdata_t[0])-9000), y_val[0,500:1500,j+3], color="r", linestyle='--', label="Target")
    ax[j][1].legend()

    if j == 0:
        ax[j][1].set_title(titles[1])
# plt.savefig("Fig/result1.pdf",bbox_inches='tight',pad_inches = 0.1)
plt.show()

j = 0
k = 0
y1 = [[],[],[]]  
y2 = [[],[],[]]  
for j in range(50):
  for k in range(3):
    y1[k].append(r2_score(Predictdata_t[j,200:,k],y_val[j,200:,k]))
    y2[k].append(r2_score(Predictdata_t[j,200:,k+3],y_val[j,200:,k+3]))
k = 0
print('R2 scores')
for k in range(3):
  print(np.mean(y1[k]))
  print(np.mean(y2[k]))

# %%
middle_layer = mymodel2.hidden_layer(x_train[:20,:,:])
print(middle_layer.shape)

# %%
def div_ev(div_list):
    d = []
    i = 0
    for i in range(100):
        d.append(div_list[0][i] - div_list[1][i])

    d = np.array(d)

    g1_w = np.sum(d[:50])
    g2_w = np.sum(d[50:100])
    sum_a = np.sum(np.abs(d))
    print((g1_w - g2_w) / sum_a)

# %%
x_batch = x_train[:50]
middle_layer_a = mymodel2.hidden_layer(x_batch)
Predictdata_a = mymodel2.predict(x_batch)

# %%
from pylab import rcParams
rcParams['figure.figsize'] = 8,8
rcParams['font.size'] = 20
rcParams['figure.dpi'] = 300
# Embed TrueType (Type 42) fonts rather than Type 3 in PDF/EPS output, so the
# figures print and embed correctly (Type 3 fonts can fail on some devices).
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
color_map = 'cividis'

middle_layer_a = tf.transpose(middle_layer_a, perm=[0, 2, 1])
den_out = tf.transpose(Predictdata_a, perm=[0, 2, 1])
mid_out = middle_layer_a.numpy()

soukan_r_list = []
temp_list_list = []
soukangyouretu_l = []
for heikin_step in range(batch_size):
    soukan = np.zeros(100*6)
    ind = 0
    temp = []
    for t_2 in range(6):
        for t_1 in range(100):
            temp = np.corrcoef(mid_out[heikin_step,t_1, 200:], den_out[heikin_step,t_2, 200:])
            soukan[ind] = temp[0,1]
            ind += 1
    soukan_r = soukan.reshape([6,100])
    soukan_r = np.abs(soukan_r)
    soukan_r_list.append(soukan_r)

    s_ave_1 = (np.abs(soukan_r[0,:]) + np.abs(soukan_r[1,:]) + np.abs(soukan_r[2,:])) / 3
    s_ave_2 = (np.abs(soukan_r[3,:]) + np.abs(soukan_r[4,:]) + np.abs(soukan_r[5,:])) / 3

    temp_list = []
    temp_list.append(s_ave_1)
    temp_list.append(s_ave_2)
    temp_list = np.array(temp_list)
    temp_list_list.append(temp_list)

    soukangyouretu = np.corrcoef(mid_out[heikin_step])
    soukangyouretu = np.abs(soukangyouretu)
    soukangyouretu_l.append(soukangyouretu)

soukan_r_list = np.array(soukan_r_list)
temp_list_list = np.array(temp_list_list)
soukan_r_abs_a = tf.reduce_mean(soukan_r_list, axis=0)
temp_list_list_a = tf.reduce_mean(temp_list_list, axis=0)
soukangyouretu_a = tf.reduce_mean(soukangyouretu_l, axis=0)

fig1_a_a, ax1_a_a = plt.subplots()
im = ax1_a_a.imshow(soukan_r_abs_a,aspect=10,cmap=color_map)
fig1_a_a.colorbar(im, ax=ax1_a_a, shrink=0.5)
plt.savefig("soukan6.pdf",bbox_inches='tight',pad_inches = 0.1)
plt.show()
plt.close('all')

print('Correlation separation index')
div_ev(temp_list)

fig2_a_a, ax2_a_a = plt.subplots()
im = ax2_a_a.imshow(temp_list_list_a, aspect=20, cmap=color_map)
fig2_a_a.colorbar(im, ax=ax2_a_a, shrink=0.34)
plt.savefig("Fig/soukan3.pdf",bbox_inches='tight',pad_inches = 0.1)
plt.show()
plt.close('all')

plt.scatter(np.abs(temp_list_list_a[0,:50]), np.abs(temp_list_list_a[1,:50]),marker="o", label="Group 1", color="b")
plt.scatter(np.abs(temp_list_list_a[0,50:]), np.abs(temp_list_list_a[1,50:]),color="r",marker="^", label="Group 2")
plt.xlabel(r"Correlation with $y^1$", fontsize=20)
plt.ylabel(r"Correlation with $y^2$", fontsize=20)
plt.legend()    
plt.savefig("Fig/per_neuron_correlation.pdf",bbox_inches='tight',pad_inches = 0.1)
plt.show()

fig2_a_s, ax2_a_s = plt.subplots()
im = ax2_a_s.imshow(soukangyouretu_a, cmap=color_map)
fig2_a_s.colorbar(im, ax=ax2_a_s, shrink=0.8)
plt.savefig("Fig/correlation_matrix.pdf",bbox_inches='tight',pad_inches = 0.1)
plt.show()
plt.close('all')

# %%
