# -*- coding: utf-8 -*-

import gcloud_utils as gcu

lines = ""

def log(line):
	global lines
	lines += line + "\n"

def save(path):
	print(lines)
	gcu.save_text(path, lines)