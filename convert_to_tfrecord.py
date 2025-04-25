import tensorflow as tf
import pandas as pd
import os
from PIL import Image

def create_tf_example(row, image_dir):
	# Lade Bild
	image_path = os.path.join(image_dir, row['filename'])
	with tf.io.gfile.GFile(image_path, 'rb') as fid:
		encoded_image = fid.read()
	image = Image.open(image_path)
	width, height = image.size

	# Bounding-Box-Koordinaten normalisieren
	xmin = [row['xmin'] / width]
	xmax = [row['xmax'] / width]
	ymin = [row['ymin'] / height]
	ymax = [row['ymax'] / height]

	# Klassen und Labels
	classes_text = [row['class'].encode('utf8')]
	classes = [1]  # 1 für "gun"

	# TFRecord-Feature erstellen
	tf_example = tf.train.Example(features=tf.train.Features(feature={
		'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image])),
		'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpg'])),
		'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
		'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
		'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
		'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
		'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
		'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
	}))
	return tf_example

def convert_csv_to_tfrecord(csv_input, image_dir, output_path):
	writer = tf.io.TFRecordWriter(output_path)
	examples = pd.read_csv(csv_input)
	for _, row in examples.iterrows():
		tf_example = create_tf_example(row, image_dir)
		writer.write(tf_example.SerializeToString())
	writer.close()
	print(f"TFRecord-Datei gespeichert: {output_path}")

# Beispielaufruf für train, valid und test mit absoluten Pfaden
convert_csv_to_tfrecord(
	'/Users/kathid/Documents/ML-models/WOD/gun-detect/train/_annotations.csv',
	'/Users/kathid/Documents/ML-models/WOD/gun-detect/train/',
	'/Users/kathid/Documents/ML-models/WOD/gun-detect/train.tfrecord'
)

convert_csv_to_tfrecord(
	'/Users/kathid/Documents/ML-models/WOD/gun-detect/valid/_annotations.csv',
	'/Users/kathid/Documents/ML-models/WOD/gun-detect/valid/',
	'/Users/kathid/Documents/ML-models/WOD/gun-detect/valid.tfrecord'
)

convert_csv_to_tfrecord(
	'/Users/kathid/Documents/ML-models/WOD/gun-detect/test/_annotations.csv',
	'/Users/kathid/Documents/ML-models/WOD/gun-detect/test/',
	'/Users/kathid/Documents/ML-models/WOD/gun-detect/test.tfrecord'
)