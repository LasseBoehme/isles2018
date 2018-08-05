# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from keras.callbacks import CSVLogger

from model import get_u_net
from isles_data import load_imgs, preprocess1
import gcloud_utils as gcu

import log



def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input")
	parser.add_argument("-o", "--output")
	args, _ = parser.parse_known_args(args=sys.argv)
	return args	


def get_kfold_indices(data, n_splits):
	kf = KFold(n_splits=n_splits)
	return kf.split(data)


def kfold_split(imgs, gts, add_data, n_splits=4, use_splits=4):
	splits = []
    
	for i, (train_indices, test_indices) in enumerate(get_kfold_indices(imgs, n_splits)):
		print("SPLIT {}".format(i))
		print("train: {}".format(train_indices))
		print("test: {}".format(test_indices))
		print("")
        
		x_train = np.concatenate([imgs[i] for i in train_indices])
		y_train = np.concatenate([gts[i] for i in train_indices])
		add_train = [add_data[i] for i in train_indices]
        
		x_test = np.concatenate([imgs[i] for i in test_indices])
		y_test = np.concatenate([gts[i] for i in test_indices])
		add_test = [add_data[i] for i in test_indices]
        
		splits.append({"x_train" : x_train, "y_train" : y_train, "add_train" : add_train, 
						 "x_test" : x_test, "y_test" : y_test, "add_test" : add_test})
    
		#if i == use_splits:
		#	break
	
	return splits


def train_model(model, kdata, model_name):	
	model.summary()
	
	for i, k in enumerate(kdata):
		m = "temp/{}_k{}.h5".format(model_name, i)
		l = "temp/{}_k{}-log.csv".format(model_name, i)
		
		earlystopper = EarlyStopping(patience=5, verbose=1)
		checkpointer = ModelCheckpoint(m, verbose=1, save_best_only=True)        
		csv_logger = CSVLogger(l, append=True, separator=";")
		
		results = model.fit(k["x_train"], k["y_train"], validation_data=(k["x_test"], k["y_test"]), batch_size=32, epochs=100, 
						callbacks=[earlystopper, checkpointer, csv_logger])

		val_dice_coef = max(results.history["val_dice_coef"])
		epochs = len(results.history["val_dice_coef"])
		log.log(m + ": {} {} epochs".format(val_dice_coef, epochs))
		
		gcu.copy(m, os.path.join(args.output, os.path.basename(m)))
		gcu.copy(l, os.path.join(args.output, os.path.basename(l)))
    
	log.log("")
		
def main():
	gcu.clear_temp()  # auf der lokalen Maschine relevant, die VM der Google Cloud wird mit jedem Job neu erstellt
	
	args = parse_arguments()
	global args		 

	imgs, gts, add_data = load_imgs(args.input, preprocess=preprocess1, size=256)

	kdata = kfold_split(imgs, gts, add_data, n_splits=4, use_splits=1)

	shape = kdata[0]["x_train"].shape[1:4]
	
	for n_fil_start in [6, 8, 10, 12, 14]:
		for d in [3, 4, 5]:
			model = get_u_net(shape, n_filter_start=n_fil_start, depth=d)
			model.summary()
			train_model(model, kdata, model_name="isles_f_{}_d{}".format(n_fil_start, d))

	log.save(os.path.join(args.output, "log1.txt"))

if __name__ == '__main__':
	main()
