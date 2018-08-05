# -*- coding: utf-8 -*-

import re
import os
import numpy as np
import nibabel as nib
from skimage import img_as_bool
from skimage.transform import resize
import gcloud_utils as gcu


def load_imgs(input_dir, size=256, preprocess=None):
	"""
	Lädt die Isles Daten auch aus dem Google Cloud Speicher
		input_dir: das Verzeichnis in dem sich die case Ordner befinden
		size: die gewünschte Kantenlänge der quadratischen Bilder
		preprocess: eine Funktion, die die geladenen Bilder vorbereitet z.B. Range Normalization (siehe aus preprocess1)
	"""
	
	PATH = "case_{}/SMIR.Brain.XX.O.{}.*/*.nii"
	CASES = range(1, 95)
	MODALITIES = ("CT", "CT_CBF", "CT_CBV",  "CT_MTT", "CT_Tmax", "OT")
	
	imgs = []
	gts = []
	additional_data = []  # [case]["case" / "id" / "header"]

	for c in CASES:
		img_mod = []
       
		for mod in MODALITIES:
			fpath = gcu.search_file(os.path.join(input_dir, PATH).format(c, mod)) 
			img = nib.load(gcu.copy_to_temp(fpath))	

			# Ausgabe der geladenen Bilder
			print("{}. {}".format(c, os.path.basename(fpath))) 
    			
			header = img.header
			img = np.transpose(img.get_data())

			# Die Bilder haben ursprünglich eine Größe von 256x256
			#if size != 256:			
			img = resize(img, (img.shape[0], size, size), mode="constant", anti_aliasing=False)
				      
			if preprocess is not None:
				img = preprocess(img, mod)
			
			if mod == "OT":
				gts.append(np.expand_dims(img_as_bool(img), axis=3))
			else:
				img_mod.append(img)

			# die img-id der CT_MTT-Datei wird bei der SMIR-Evaluation für für die Identifikation des cases verwendet
			if mod == "CT_MTT":
				num = re.findall(r"\d+", fpath)
				additional_data.append({"case" : num[1], "id" : num[3], "header" : header})

		imgs.append(np.stack(img_mod, axis=3))

	print("")
	return imgs, gts, additional_data


def preprocess1(img, mod):
	"""
	Preprocess-Funktionen müssen diese beiden Argumente haben:
		img: das Bild als 3d-numpy array
		mod: die Modalität (falls das Preprocessing da unterschiedlch ausfallen soll)
	
	und muss das vorbereitete Bild zurückgeben
	"""
	
	# Minuswerte zu 0 (im CT negative Hounsfield-Units für Fett, Luft als 0 definiert)
	img[img < 0] = 0  
    			
	# Density-Range von 0-1    
	img /= img.max()

	return img       
    
 			     