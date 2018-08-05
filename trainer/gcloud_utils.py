# -*- coding: utf-8 -*-


"""
Die Programme aus der Google Machine Learning Cloud laden und speichern die Daten aus einem Cloud Storage auf den
mithilf der normalen Python IO-Funktionen nicht zugegriffen werden kann. Allerdings kann z.B. nibabel.load() nur 
auf das lokale Dateisystem zurückgreifen. Die Dateien müssen also zuerst vom Cloud Storage auf die Maschine (auch in der Cloud)
kopiert werden und können dann von dort gelesen werden.
"""


import os
from tensorflow.python.lib.io import file_io

TEMP_PATH = "./temp"

def get_temp():		
	if not os.path.exists(TEMP_PATH):
		os.makedirs(TEMP_PATH)
	
	return TEMP_PATH


def clear_temp():
	file_io.delete_recursively(get_temp())


def copy(src, dest):
	file_data = file_io.FileIO(src, "r").read()	
	file_io.FileIO(dest, "w").write(file_data) 
    

def copy_to_temp(src):     
	dest_path = os.path.join(get_temp(), os.path.basename(src))
    
	copy(src, dest_path)
    
	return dest_path
	

def save_text(path, text):
	file_io.FileIO(path, "w").write(text)


def search_file(path):
	file_path = file_io.get_matching_files(path)[0]  # file_io kann auch auf dem Cloud Speicher suchen
	return file_path