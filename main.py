

# подключаем библиотеки


import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sqlite3

# определяем функцию для приведения данных в определенный формат
def make_data_uni(data, start, end, size_of_hist, size):
  _data = []
  labels = []

  start = start + size_of_hist
  if end is None:
    end = len(data) - size

  for i in range(start, end):
    indices = range(i - size_of_hist, i)

    _data.append(np.reshape(data[indices], (size_of_hist, 1)))
    labels.append(data[i + size])
  return np.array(_data), np.array(labels)
# определяем интервалы
def create_time_intervals(length):
  return list(range(-length, 0))
# определяем функцию для визуализации расхождения данных фактически и предугаданных
def show_diff_plot(plot_data, delta, title):
  labels = ['История', 'Истинное значение', 'Предугаданное значение']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_intervals(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Временные интервалы')
  return plt

def baseline(history):
  return np.mean(history)
# определяем функцию для приведения данных в определенный формат для несколькоих признаков
def make_data_multi(data, target, start, end, size_hist, size, step, single_step=False):
  _data = []
  labels = []

  start = start + size_hist
  if end is None:
    end = len(data) - size

  for i in range(start, end):
    indices = range(i - size_hist, i, step)
    _data.append(data[indices])

    if single_step:
      labels.append(target[i + size])
    else:
      labels.append(target[i:i+size])

  return np.array(_data), np.array(labels)
# определяем функцию для визуализации
def plot_hist_training(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Величина ошибки на стадии обучения')
  plt.plot(epochs, val_loss, 'r', label='Величина ошибки на стадии тестирования')
  plt.title(title)
  plt.legend()

  plt.show()
# определяем функцию для визуализации
def plot_multi_step(history, true_future, prediction, csv_file):
  pd.DataFrame(data={"Предугаданное значение": prediction, "Реальное значение": true_future}).to_csv(csv_file, sep=',', index=False)

  plt.figure(figsize=(12, 6))
  num_in = create_time_intervals(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 1]), label='История')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo', label='Истинное значение')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro', label='Предугаданное знчение')
  plt.legend(loc='upper left')
  plt.show()
# настраиваем внешний вид графиков
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False



try:
    connection = sqlite3.connect('/Users/aleksandrgrigorev/Desktop/DATA FOR AI/DB.db')
    cursor = connection.cursor()

    df = pd.read_sql_query("select DATETIME,BRENT,GOLD,EURUSD from dataset", connection)
except sqlite3.Error as error:
    print("Ошибка при подключении к sqlite", error)
finally:
    if (connection):
        connection.close()


#print(df.head())

TRAIN_SPLIT = 100000
tf.random.set_seed(20)

'''
uni_data = df['VALUE']
uni_data.index = df['DATETIME']
#print(uni_data.head())
#uni_data.plot(subplots=True)
uni_data = uni_data.values
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()
uni_data = (uni_data-uni_train_mean)/uni_train_std

univariate_PASTHISTORY = 20
univariate_FUTURETARGET = 0
x_train_uni, y_train_uni = make_data_uni(uni_data, 0, TRAIN_SPLIT,
                                           univariate_PASTHISTORY,
                                           univariate_FUTURETARGET)
x_val_uni, y_val_uni = make_data_uni(uni_data, TRAIN_SPLIT, None,
                                       univariate_PASTHISTORY,
                                       univariate_FUTURETARGET)

#print ('Single window of past history')
#print (x_train_uni[0])
#print ('\n Target temperature to predict')
#print (y_train_uni[0])

#show_diff_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')

#show_diff_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0, 'Baseline Prediction Example')
'''

BATCHSIZE = 50

BUFFERSIZE = 100000
EVALUATIONINTERVAL = 40
ROUNDS = 5



'''
train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFERSIZE).batch(BATCHSIZE).repeat()
val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCHSIZE).repeat()
simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])
simple_lstm_model.compile(optimizer='adam', loss='mae')
for x, y in val_univariate.take(1):
    print(simple_lstm_model.predict(x).shape)

simple_lstm_model.fit(train_univariate, epochs=ROUNDS,
                      steps_per_epoch=EVALUATIONINTERVAL,
                      validation_data=val_univariate, validation_steps=50)
#simple_lstm_model.save()
for x, y in val_univariate.take(10):
  plot = show_diff_plot([x[0].numpy(), y[0].numpy(),
                    simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
  plot.show()
'''


features_considered = [
    'BRENT',
    'GOLD',
    'EURUSD'
]
features = df[features_considered]
features.index = df['DATETIME']
#features.head()
#features.plot(subplots=True)

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset-data_mean)/data_std
PASTHISTORY = 720
FUTURETARGET = 72
STEP = 2

x_train_single, y_train_single = make_data_multi(dataset, dataset[:, 1], 0,
                                                   TRAIN_SPLIT, PASTHISTORY,
                                                   FUTURETARGET, STEP,
                                                   single_step=True)
x_val_single, y_val_single = make_data_multi(dataset, dataset[:, 1],
                                               TRAIN_SPLIT, None, PASTHISTORY,
                                               FUTURETARGET, STEP,
                                               single_step=True)
#print ('Single window of past history : {}'.format(x_train_single[0].shape))
train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFERSIZE).batch(BATCHSIZE).repeat()
val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCHSIZE).repeat()
single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32, input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))
single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
#for x, y in val_data_single.take(1):
#  print(single_step_model.predict(x).shape)
single_step_history = single_step_model.fit(train_data_single, epochs=ROUNDS,
                                            steps_per_epoch=EVALUATIONINTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=100)
#plot_hist_training(single_step_history, 'Single Step Training and validation loss')
for x, y in val_data_single.take(10):
  plot = show_diff_plot([x[0][:, 1].numpy(), y[0].numpy(),
                    single_step_model.predict(x)[0]], 12,
                   'Точечное предугадывание')
  plot.show()



#multi-parameters
FUTURETARGET = 72
x_train_multi, y_train_multi = make_data_multi(dataset, dataset[:, 1], 0,
                                                 TRAIN_SPLIT, PASTHISTORY,
                                                 FUTURETARGET, STEP)
x_val_multi, y_val_multi = make_data_multi(dataset, dataset[:, 1],
                                             TRAIN_SPLIT, None, PASTHISTORY,
                                             FUTURETARGET, STEP)
#print ('Single window of past history : {}'.format(x_train_multi[0].shape))
#print ('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))
train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFERSIZE).batch(BATCHSIZE).repeat()
val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCHSIZE).repeat()
#for x, y in train_data_multi.take(1):
#  plot_multi_step(x[0], y[0], np.array([0]))
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(72))
multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
#for x, y in val_data_multi.take(1):
#  print (multi_step_model.predict(x).shape)
multi_step_history = multi_step_model.fit(train_data_multi, epochs=ROUNDS,
                                          steps_per_epoch=EVALUATIONINTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)
plot_hist_training(multi_step_history, 'Ошибки обучения и тестирования тренда')
i = 1
for x, y in val_data_multi.take(3):
  plot_multi_step(x[0], y[0], multi_step_model.predict(x)[0], './export' + str(i) + '.csv')
  i = i + 1

print(1)