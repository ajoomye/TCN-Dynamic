import csv, random, json
import pandas as pd
import numpy as np
from tensorflow import keras
from keras import backend as K
from keras.optimizers import Optimizer
import tensorflow as tf
from timeit import default_timer as timer
from sklearn.metrics import classification_report
from  sklearn.metrics import precision_recall_fscore_support
from keras.models import Sequential
from keras.layers import Conv1D, ReLU, Dropout, Add, Embedding, Dense, Input, GlobalAveragePooling1D, Layer, Softmax
import os

import numpy as np
import tensorflow as tf
import random


##### IMPORTANT #####
#COPY FILES FROM EXTRACTED_DATA TO THE SAME DIRECTORY AS THIS FILE

### FOR ORIGINAL CIC20 DATASET WITH TXT AND BINDERS
### AVERAGE LENGTH SYSCALLS: 47411
### NUMBER OF DIFFERENT SYSCALLS: 140
### AVERAGE LENGTH BINDERS: 374
### NUMBER OF DIFFERENT BINDERS: 298
### NUMBER OF DIFFERENT SYSCALLS for 1000: 140
### NUMBER OF DIFFERENT BEHAVIOURS: 34
### AVERAGE LENGTH Comp Behaviour: 520




allresultstcn={}
allresultslstm={}
def training(size, kernel_size, dilation, filter_size, embed_size, batch_size):
	input_malware_file_train = f"all_{size}_malware_syscalls_CIC20ori_token_train.csv"
	with open(input_malware_file_train) as input_malware_train:
		reader_malware_train = csv.reader(input_malware_train)
		all_malware_syscalls_train = list(reader_malware_train)

	input_benign_file_train = f"all_{size}_benign_syscalls_CIC20ori_token_train.csv"
	with open(input_benign_file_train) as input_benign_train:
		reader_benign_train = csv.reader(input_benign_train)
		all_benign_syscalls_train = list(reader_benign_train)

	input_malware_file_test = f"all_{size}_malware_syscalls_CIC20ori_token_test.csv"
	with open(input_malware_file_test) as input_malware_test:
		reader_malware_test = csv.reader(input_malware_test)
		all_malware_syscalls_test = list(reader_malware_test)

	input_benign_file_test = f"all_{size}_benign_syscalls_CIC20ori_token_test.csv"
	with open(input_benign_file_test) as input_benign_test:
		reader_benign_test = csv.reader(input_benign_test)
		all_benign_syscalls_test = list(reader_benign_test)


	###BINDERS FILES

	input_malware_file_train = f"all_totalavg_malware_binders_CIC20ori_token_train.csv"
	with open(input_malware_file_train) as input_malware_train:
		reader_malware_train = csv.reader(input_malware_train)
		all_malware_binders_train = list(reader_malware_train)

	input_benign_file_train = f"all_totalavg_benign_binders_CIC20ori_token_train.csv"
	with open(input_benign_file_train) as input_benign_train:
		reader_benign_train = csv.reader(input_benign_train)
		all_benign_binders_train = list(reader_benign_train)

	input_malware_file_test = f"all_totalavg_malware_binders_CIC20ori_token_test.csv"
	with open(input_malware_file_test) as input_malware_test:
		reader_malware_test = csv.reader(input_malware_test)
		all_malware_binders_test = list(reader_malware_test)

	input_benign_file_test = f"all_totalavg_benign_binders_CIC20ori_token_test.csv"
	with open(input_benign_file_test) as input_benign_test:
		reader_benign_test = csv.reader(input_benign_test)
		all_benign_binders_test = list(reader_benign_test)

	###COMPBEHAVIOUR FILES

	input_malware_file_train = f"all_totalavg_malware_compbehaviour_CIC20ori_token_train.csv"
	with open(input_malware_file_train) as input_malware_train:
		reader_malware_train = csv.reader(input_malware_train)
		all_malware_compbehaviour_train = list(reader_malware_train)

	input_benign_file_train = f"all_totalavg_benign_compbehaviour_CIC20ori_token_train.csv"
	with open(input_benign_file_train) as input_benign_train:
		reader_benign_train = csv.reader(input_benign_train)
		all_benign_compbehaviour_train = list(reader_benign_train)

	input_malware_file_test = f"all_totalavg_malware_compbehaviour_CIC20ori_token_test.csv"
	with open(input_malware_file_test) as input_malware_test:
		reader_malware_test = csv.reader(input_malware_test)
		all_malware_compbehaviour_test = list(reader_malware_test)

	input_benign_file_test = f"all_totalavg_benign_compbehaviour_CIC20ori_token_test.csv"
	with open(input_benign_file_test) as input_benign_test:
		reader_benign_test = csv.reader(input_benign_test)
		all_benign_compbehaviour_test = list(reader_benign_test)


	###SYSCALLS AND LABELS
	all_data_train_syscalls = all_malware_syscalls_train + all_benign_syscalls_train
	all_data_test_syscalls = all_malware_syscalls_test + all_benign_syscalls_test

	all_data_train_binders = all_malware_binders_train + all_benign_binders_train
	all_data_test_binders = all_malware_binders_test + all_benign_binders_test

	all_data_train_compbehaviour = all_malware_compbehaviour_train + all_benign_compbehaviour_train
	all_data_test_compbehaviour = all_malware_compbehaviour_test + all_benign_compbehaviour_test

	all_train_data = []

	for i in range(len(all_data_train_syscalls)):
		train_syscalls = all_data_train_syscalls[i]
		train_binders = all_data_train_binders[i]
		train_compbehaviours = all_data_train_compbehaviour[i]
		#print(len(train_binders))

		# Combine the elements from A and B (excluding their last element) and include the class label at the end
		combined_item = [train_syscalls[:-1], train_binders[:-1], train_compbehaviours[:-1], train_syscalls[-1]]
		all_train_data.append(combined_item)

	all_test_data = []

	for i in range(len(all_data_test_syscalls)):
		test_syscalls = all_data_test_syscalls[i]
		test_binders = all_data_test_binders[i]
		test_compbehaviours = all_data_test_compbehaviour[i]

		# Combine the elements from A and B (excluding their last element) and include the class label at the end
		combined_item = [test_syscalls[:-1], test_binders[:-1], test_compbehaviours[:-1], test_syscalls[-1]]
		all_test_data.append(combined_item)

	random.shuffle(all_train_data)
	random.shuffle(all_test_data)

	all_syscalls_train = []
	all_binders_train = []
	all_compehaviours_train = []
	all_labels_train = []
	for i in all_train_data:
		all_syscalls_train.append(i[0])
		all_binders_train.append(i[1])
		all_compehaviours_train.append(i[2])
		all_labels_train.append(i[3])

	all_syscalls_test = []
	all_binders_test = []
	all_compehaviours_test = []
	all_labels_test = []
	for i in all_test_data:
		all_syscalls_test.append(i[0])
		all_binders_test.append(i[1])
		all_compehaviours_test.append(i[2])
		all_labels_test.append(i[3])

	#print(all_binders_train)

	all_syscalls_train_np = np.array(all_syscalls_train, dtype=int)
	all_labels_train_np = np.array(all_labels_train, dtype=int)

	all_syscalls_test_np = np.array(all_syscalls_test, dtype=int)
	all_labels_test_np = np.array(all_labels_test, dtype=int)

	all_binders_train_np = np.array(all_binders_train, dtype=int)
	all_binders_test_np = np.array(all_binders_test, dtype=int)

	all_compehaviours_train_np = np.array(all_compehaviours_train, dtype=int)
	all_compehaviours_test_np = np.array(all_compehaviours_test, dtype=int)



	print(len(all_compehaviours_test_np))


		

	input_syscalls = Input(shape=(syscall_size,))
	input_binders = Input(shape=(374,))
	input_compbehaviours = Input(shape=(520,))


	#SYSCALLS MODEL
	model_s = keras.layers.Embedding(140, 32, input_length=syscall_size)(input_syscalls)
	sys_dilation = [1,2,4,8,16]
	for dil in sys_dilation:
		model_s = keras.layers.Conv1D(filters=64,padding='causal', kernel_size=32, dilation_rate=dil, activation='relu')(model_s)

	model_s = keras.layers.GlobalAveragePooling1D()(model_s)
	#model_s = keras.layers.Flatten()(model_s)
	model_s = keras.models.Model(inputs=input_syscalls, outputs=model_s)


	#binders MODEL
	model_p = keras.layers.Embedding(298, 32, input_length=374)(input_binders)
	prot_dilation = [1,2,4,8,16,32]
	for dil in prot_dilation:
		model_p = keras.layers.Conv1D(filters=64, padding='causal', kernel_size=8,
										dilation_rate=dil, activation='relu')(model_p)
	model_p = keras.layers.GlobalAveragePooling1D()(model_p)
	#model_p = keras.layers.Flatten()(model_p)
	model_p = keras.models.Model(inputs=input_binders, outputs=model_p)

	# Compbehaviours MODEL
	model_c = keras.layers.Embedding(34, 32, input_length=520)(input_compbehaviours)
	comp_dilation = [1,2,4,8,16,32]
	for dil in comp_dilation:
		model_c = keras.layers.Conv1D(filters=64, padding='causal', kernel_size=8,
										dilation_rate=dil, activation='relu')(model_c)
	model_c = keras.layers.GlobalAveragePooling1D()(model_c)
	# model_p = keras.layers.Flatten()(model_p)
	model_c = keras.models.Model(inputs=input_compbehaviours, outputs=model_c)



	combined = keras.layers.concatenate([model_s.output, model_p.output, model_c.output])
	output_layer = Dense(1, activation='sigmoid')(combined)

	model = keras.models.Model(inputs=[model_s.input, model_p.input, model_c.input], outputs=output_layer)




	class TimingCallback(keras.callbacks.Callback):
		def __init__(self, logs={}):
			self.logs = []

		def on_epoch_begin(self, epoch, logs={}):
			self.starttime = timer()

		def on_epoch_end(self, epoch, logs={}):
			self.logs.append(timer() - self.starttime)

	cb = TimingCallback()

	
	model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives(),tf.keras.metrics.FalseNegatives(),tf.keras.metrics.FalsePositives()])


	
	model.summary()
	#keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	recordname = f"final_model"
	#keras.utils.plot_model(model, to_file=f'{recordname}.png')
	csv_name = f'{recordname}_results.csv'
	csv_logger = tf.keras.callbacks.CSVLogger(csv_name)
	history = model.fit([all_syscalls_train_np, all_binders_train_np, all_compehaviours_train_np], all_labels_train_np, validation_split=0.125, epochs=300, batch_size=batch_size, callbacks=[csv_logger,cb])
	#history = model.fit([all_syscalls_train_np, all_binders_train_np, all_compehaviours_train_np], all_labels_train_np, validation_data=([all_syscalls_test_np, all_binders_test_np, all_compehaviours_test_np], all_labels_test_np), epochs=300, batch_size=batch_size, callbacks=[csv_logger, cb])
	model.save(f"{recordname}_weights.h5")

	read_csv= pd.read_csv(csv_name)
	read_csv['traintime']=cb.logs
	read_csv.to_csv(csv_name)

	print(cb.logs)
	print(sum(cb.logs))



'''
The training function takes in the following parameters:
1. The size for the system calls sequences
2. The kernel size for the convolutional layers
3. The dilation rate for the convolutional layers
4. The filter size for the convolutional layers
5. The embedding size for the embedding layers
6. The batch size for training

'''


training(10000, 8, [1,2,4,8,16,32], 64, 32, 32)
