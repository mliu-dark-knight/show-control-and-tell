import argparse
import itertools
import os
import pickle

import h5py
import numpy as np
from speaksee.data import DictionaryDataset, Dataset, RawField
from speaksee.data import TextField, ImageDetectionsField
from tqdm import tqdm

from config import *
from data import COCOControlSetField
from data.dataset import COCOEntities

n_pkl_entries = 5


def count_total_n_detections(key_dataset: Dataset, hdf5_input):
	n_detections = 0
	for image_index in tqdm(range(len(key_dataset)), ncols=100):
		image = key_dataset.examples[image_index].image
		id_image = int(image.split('/')[-1].split('_')[-1].split('.')[0])

		det_cls_probs = hdf5_input['%s_cls_prob' % id_image][()]
		non_background_idxs = np.squeeze(np.argwhere(np.argmax(det_cls_probs, -1) != 0), axis=1)
		n_detections += len(non_background_idxs)
	return n_detections


def retrieve_boxes(cls_seq, selected_classes, sorted_idxs, max_detections):
	bboxes_seq = []
	for cls in cls_seq:
		if cls is None:
			bboxes_seq.append([])
		elif cls == '_':
			bboxes_seq.append(list(sorted_idxs[:max_detections]))
		else:
			normalized_cls = normalize_cls(cls)
			seed_detections = [i for i, c in enumerate(selected_classes) if c == cls or c == normalized_cls]
			if len(seed_detections) > 0:
				bboxes_seq.append(list(np.unique(seed_detections)[:max_detections]))
			# avoid empty boxes
			else:
				bboxes_seq.append(list(sorted_idxs[:max_detections]))
	assert len(list(itertools.chain.from_iterable(bboxes_seq))) > 0
	return bboxes_seq


def normalize_cls(cls):
	return cls.split(',')[0].split(' ')[-1]


def get_file_name(file_path):
	return file_path.split('/')[-1]


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='coco', type=str, help='dataset: coco | flickr')
	parser.add_argument('--output-pkl-file', default='prepro_data/coco-captioning-bottomup.pkl')
	parser.add_argument('--output-hdf5-file', default='prepro_data/coco-captioning-bottomup.h5')
	parser.add_argument('--output-glove-file', default='prepro_data/captioning-glove.pkl')
	return parser.parse_args()


if __name__ == '__main__':
	opt = parse_args()

	image_field = ImageDetectionsField(detections_path=os.path.join(coco_root, 'coco_detections.hdf5'),
	                                   load_in_tmp=False)

	det_field = COCOControlSetField(detections_path=os.path.join(coco_root, 'coco_detections.hdf5'),
	                                classes_path=os.path.join(coco_root, 'object_class_list.txt'),
	                                img_shapes_path=os.path.join(coco_root, 'coco_img_shapes.json'),
	                                precomp_glove_path=os.path.join(coco_root, 'object_class_glove.pkl'),
	                                max_detections=20)

	text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, remove_punctuation=True, fix_length=20)

	dataset = COCOEntities(image_field, det_field, RawField(),
	                       img_root='',
	                       ann_root=os.path.join(coco_root, 'annotations'),
	                       entities_file=os.path.join(coco_root, 'coco_entities.json'),
	                       id_root=os.path.join(coco_root, 'annotations'),
	                       filtering=True)

	example_split_map = {}
	for examples, split in zip([dataset.train_examples, dataset.val_examples, dataset.test_examples],
	                           ['train', 'val', 'test']):
		for example in examples:
			image_path = get_file_name(example.image)
			assert image_path not in example_split_map or split == example_split_map[image_path]
			example_split_map[image_path] = split

	# TODO: debug mode
	# random_idxs = np.random.choice(len(dataset), int(len(dataset) / 5000), replace=False)
	# dataset = DictionaryDataset([dataset.examples[idx] for idx in random_idxs], dataset.fields, 'image')
	dataset = DictionaryDataset(dataset.examples, dataset.fields, 'image')
	key_dataset, value_dataset = dataset.key_dataset, dataset.value_dataset

	hdf5_input = h5py.File(det_field.detections_path, 'r')
	total_n_detections = count_total_n_detections(key_dataset, hdf5_input)

	# TODO: do not forget to close
	hdf5_output = h5py.File(opt.output_hdf5_file, 'w')
	features = hdf5_output.create_dataset('features', shape=(total_n_detections, 2048), dtype=np.float32)
	cls_labels = hdf5_output.create_dataset('cls_labels', shape=(total_n_detections,), dtype=int)
	spatial = hdf5_output.create_dataset('spatial', shape=(total_n_detections, 4), dtype=np.float32)
	boxes = hdf5_output.create_dataset('boxes', shape=(total_n_detections, 4), dtype=int)

	output_pkl = {'train': [[] for _ in range(n_pkl_entries)],
	              'val': [[] for _ in range(n_pkl_entries)],
	              'test': [[] for _ in range(n_pkl_entries)]}
	n_detections = 0

	for image_index in tqdm(range(len(key_dataset)), ncols=100):
		img_file_path = get_file_name(key_dataset.examples[image_index].image)
		id_image = int(img_file_path.split('_')[-1].split('.')[0])
		width, height = det_field.img_shapes[str(id_image)]

		det_cls_probs = hdf5_input['%s_cls_prob' % id_image][()]
		det_features = hdf5_input['%s_features' % id_image][()]
		det_boxes = hdf5_input['%s_boxes' % id_image][()]
		assert len(det_cls_probs) == len(det_features) and len(det_features) == len(det_boxes)
		non_background_idxs = np.squeeze(np.argwhere(np.argmax(det_cls_probs, -1) != 0), axis=1)
		det_cls_probs = det_cls_probs[non_background_idxs]
		det_features = det_features[non_background_idxs]
		det_boxes = det_boxes[non_background_idxs]
		sorted_idxs = np.argsort(np.max(det_cls_probs, -1))[::-1]
		sorted_dets = det_features[sorted_idxs]

		img_n_detection = len(det_boxes)
		features[n_detections: n_detections + img_n_detection] = det_features
		cls_labels[n_detections: n_detections + img_n_detection] = np.argmax(det_cls_probs, axis=-1)
		boxes[n_detections: n_detections + img_n_detection] = det_boxes
		# TODO: may need to modify spatial feature
		assert np.all(det_boxes[:, [0, 2]] <= width)
		assert np.all(det_boxes[:, [1, 3]] <= height)
		spatial[n_detections: n_detections + img_n_detection] \
			= det_boxes / np.expand_dims([width, height, width, height], axis=0)
		hdf5_output.flush()
		n_detections += img_n_detection

		selected_classes = [det_field.classes[np.argmax(det_cls_probs[i])] for i in range(len(det_cls_probs))]
		split = example_split_map[img_file_path]

		for caption_index in value_dataset.dictionary[image_index]:
			cls_seq = value_dataset.examples[caption_index].detection[1]
			text = value_dataset.examples[caption_index].text
			if len(text.split()) != cls_seq:
				continue
			bboxes_seq = retrieve_boxes(cls_seq, selected_classes, sorted_idxs, det_field.max_detections)
			output_pkl[split][0].append(text)
			output_pkl[split][1].append([i for i in range(n_detections - img_n_detection, n_detections, 1)])
			output_pkl[split][2].append(list(cls_seq))
			output_pkl[split][3].append(bboxes_seq)
			output_pkl[split][4].append(img_file_path)

	assert n_detections == total_n_detections
	hdf5_output.close()

	output_glove = {'classes': det_field.classes, 'vectors': det_field.vectors}
	with open(opt.output_pkl_file, 'wb') as f:
		pickle.dump(output_pkl, f)
	with open(opt.output_glove_file, 'wb') as f:
		pickle.dump(output_glove, f)
