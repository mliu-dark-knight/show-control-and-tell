import argparse
import os
import pickle

import h5py
import numpy as np
from speaksee.data import DictionaryDataset, Dataset
from speaksee.data import TextField, RawField

from config import *
from data import FlickrDetectionField, FlickrControlSetField
from data.dataset import FlickrEntities

n_pkl_entries = 5


def count_total_n_detections(key_dataset: Dataset, hdf5_input):
	n_detections = 0
	for image_index in range(len(key_dataset)):
		image = key_dataset.examples[image_index].image
		id_image = int(image.split('/')[-1].split('_')[-1].split('.')[0])

		det_cls_probs = hdf5_input['%s_cls_prob' % id_image][()]
		non_background_idxs = np.squeeze(np.argwhere(np.argmax(det_cls_probs, -1) != 0), axis=1)
		n_detections += len(non_background_idxs)
	return n_detections


def retrieve_boxes(cls_seq, gt_bboxes, det_bboxes):
	bboxes_seq = []
	for cls in cls_seq:
		if cls is None:
			bboxes_seq.append([])
		else:
			id_boxes = set()
			for k, bbox in enumerate(gt_bboxes[cls]):
				id_bbox = -1
				iou_max = 0
				for ii, det_bbox in enumerate(det_bboxes):
					iou = FlickrControlSetField._bb_intersection_over_union(bbox, det_bbox)
					if iou_max < iou:
						id_bbox = ii
						iou_max = iou
				assert id_bbox > -1
				id_boxes.add(id_bbox)
			bboxes_seq.append(list(id_boxes))
	return bboxes_seq


def get_file_name(file_path):
	return file_path.split('/')[-1]


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--output-pkl-file', default='flickr-captioning-bottomup.pkl')
	parser.add_argument('--output-hdf5-file', default='flickr-captioning-bottomup.h5')
	parser.add_argument('--output-glove-file', default='captioning-glove.pkl')
	return parser.parse_args()


if __name__ == '__main__':
	opt = parse_args()

	image_field = FlickrDetectionField(detections_path=os.path.join(flickr_root, 'flickr30k_detections.hdf5'))

	det_field = FlickrControlSetField(detections_path=os.path.join(flickr_root, 'flickr30k_detections.hdf5'),
	                                  classes_path=os.path.join(flickr_root, 'object_class_list.txt'),
	                                  img_shapes_path=os.path.join(flickr_root, 'flickr_img_shapes.json'),
	                                  precomp_glove_path=os.path.join(flickr_root, 'object_class_glove.pkl'))

	text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, remove_punctuation=True, fix_length=20)

	dataset = FlickrEntities(image_field, RawField(), det_field,
	                         img_root='',
	                         ann_file=os.path.join(flickr_root, 'flickr30k_annotations.json'),
	                         entities_root=flickr_entities_root)

	example_split_map = {}
	for examples, split in zip([dataset.train_examples, dataset.val_examples, dataset.test_examples],
	                           ['train', 'val', 'test']):
		for example in examples:
			image_path = get_file_name(example.image)
			assert image_path not in example_split_map or split == example_split_map[image_path]
			example_split_map[image_path] = split

	# TODO: debug mode
	max_index = int(len(dataset) / 5000)
	dataset = DictionaryDataset(dataset.examples[:max_index], dataset.fields, 'image')
	# dataset = DictionaryDataset(dataset.examples, dataset.fields, 'image')
	key_dataset, value_dataset = dataset.key_dataset, dataset.value_dataset

	hdf5_input = h5py.File(det_field.detections_path, 'r')
	total_n_detections = count_total_n_detections(key_dataset, hdf5_input)

	# TODO: do not forget to close
	hdf5_output = h5py.File(opt.output_hdf5_file, 'w')
	features = hdf5_output.create_dataset('features', shape=(total_n_detections, 2048), dtype=np.float32)
	cls_probs = hdf5_output.create_dataset('cls_prob', shape=(total_n_detections, 1601), dtype=int)
	spatial = hdf5_output.create_dataset('spatial', shape=(total_n_detections, 4), dtype=np.float32)
	boxes = hdf5_output.create_dataset('boxes', shape=(total_n_detections, 4), dtype=int)

	output_pkl = {'train': [[] for _ in range(n_pkl_entries)],
	              'val': [[] for _ in range(n_pkl_entries)],
	              'test': [[] for _ in range(n_pkl_entries)]}
	n_detections = 0

	for image_index in range(len(key_dataset)):
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
		cls_probs[n_detections: n_detections + img_n_detection] = det_cls_probs
		boxes[n_detections: n_detections + img_n_detection] = det_boxes
		# TODO: may need to modify spatial feature
		spatial[n_detections: n_detections + img_n_detection] \
			= det_boxes / np.expand_dims([width, height, width, height], axis=0)
		hdf5_output.flush()
		n_detections += img_n_detection

		selected_classes = [det_field.classes[np.argmax(det_cls_probs[i])] for i in range(len(det_cls_probs))]
		split = example_split_map[img_file_path]

		for caption_index in value_dataset.dictionary[image_index]:
			x = value_dataset.examples[caption_index].detection
			gt_bboxes = x[1]
			cls_seq = [d - 1 if d > 0 else None for d in x[2]]
			bboxes_seq = retrieve_boxes(cls_seq, gt_bboxes, det_boxes)
			# original caption
			output_pkl[split][0].append(value_dataset.examples[caption_index].text)
			# set of all objects idxs
			output_pkl[split][1].append([i for i in range(n_detections - img_n_detection, n_detections, 1)])
			# sequence of chunks
			output_pkl[split][2].append(cls_seq)
			# sequence of object idxs
			output_pkl[split][3].append(bboxes_seq)
			# image file name
			output_pkl[split][4].append(img_file_path)

	assert n_detections == total_n_detections
	hdf5_output.close()

	output_glove = {'classes': det_field.classes, 'vectors': det_field.vectors}
	with open(opt.output_pkl_file, 'wb') as f:
		pickle.dump(output_pkl, f)
	with open(opt.output_glove_file, 'wb') as f:
		pickle.dump(output_glove, f)
