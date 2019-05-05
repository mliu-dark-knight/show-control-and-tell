import argparse
import pickle

import h5py


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--output-pkl-file', default='coco-captioning-bottomup.pkl')
	parser.add_argument('--output-hdf5-file', default='coco-captioning-bottomup.h5')
	parser.add_argument('--output-glove-file', default='captioning-glove.pkl')
	return parser.parse_args()


if __name__ == '__main__':
	opt = parse_args()
	hdf5_input = h5py.File(opt.output_hdf5_file, 'r')
	features = hdf5_input['features']
	cls_probs = hdf5_input['cls_prob']
	spatial = hdf5_input['spatial']
	boxes = hdf5_input['boxes']

	with open(opt.output_pkl_file, 'rb') as fp:
		dataset = pickle.load(fp)
	with open(opt.output_glove_file, 'rb') as fp:
		glove = pickle.load(fp)
