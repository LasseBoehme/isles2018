# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from loss_functions import dice_coef, dice_coef_loss


def get_u_net(input_shape, n_filter_start=8, depth=5, filter_size=3, pooling_size=2):
	n_filter = n_filter_start

	# ll = last_layer
	inputs = Input(input_shape)
	ll = inputs
    
	# von diesen Convolution-Layers wird auf die Upconvolution-Layers kopiert
	copy_layers = []
    
	# Downconvolution
	for i in range(0, depth-1):
		for i in range(0, 2):
			ll = Conv2D(n_filter, (filter_size, filter_size), activation='relu', padding='same') (ll)
        
		copy_layers.append(ll)
		ll = MaxPooling2D((pooling_size, pooling_size)) (ll)
		n_filter *= 2
        
	ll = Conv2D(n_filter, (filter_size, filter_size), activation='relu', padding='same') (ll)
	ll = Conv2D(n_filter, (filter_size, filter_size), activation='relu', padding='same') (ll)

	# Upconvolution
	for i in range(0, depth-1):
		n_filter //= 2
        
		ll = Conv2DTranspose(n_filter, (pooling_size, pooling_size), strides=(pooling_size, pooling_size), padding='same') (ll)
		ll = concatenate([ll, copy_layers.pop()])
        
		for i in range(0, 2):
			ll = Conv2D(n_filter, (filter_size, filter_size), activation='relu', padding='same') (ll)
    
	outputs = Conv2D(1, (1, 1), activation='sigmoid') (ll)

	model = Model(inputs=[inputs], outputs=[outputs])
	model.compile(optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef])
    
	return model
