import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from model import get_unet
from isles_images import load_imgs
from sklearn.model_selection import train_test_split
from tensorflow.python.lib.io import file_io
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input")
	parser.add_argument("-o", "--output")
	args, _ = parser.parse_known_args(args=sys.argv)
	return args	


def main():
	args = parse_arguments()
	
	imgs, gts, additional_data = load_imgs(args.input, 128)

	x_train, x_test_case, y_train, y_test_case, add_train, add_test = \
    train_test_split(imgs, gts, additional_data, shuffle=False, test_size=0.25)

	x_train = np.concatenate(x_train, axis=0)
	y_train = np.concatenate(y_train, axis=0)
	x_test = np.concatenate(x_test_case, axis=0)
	y_test = np.concatenate(y_test_case, axis=0)

	model = get_unet(x_train)
	
	earlystopper = EarlyStopping(patience=5, verbose=1)
	checkpointer = ModelCheckpoint("unet-gcloud.hd5f", verbose=1, save_best_only=True)

	results = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=150, callbacks=[earlystopper, checkpointer])
	
	# Workaround
	file_data = file_io.FileIO("unet-gcloud.hd5f", "r").read()	
	file_io.FileIO(os.path.join(args.output, "unet-gcloud.hd5f"), "w").write(file_data)

if __name__ == '__main__':
	main()

