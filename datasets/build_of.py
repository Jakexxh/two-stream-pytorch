from __future__ import print_function

import os
import sys
import glob
import argparse
import cv2
import skvideo.io
import scipy.misc
import numpy as np
from PIL import Image
from pipes import quote
from multiprocessing import Pool, current_process


def dense_flow(video_path, video_name, image_path, flow_x_path, flow_y_path):
	'''
	To extract dense_flow images
	:param augs:the detailed augments:
		video_name: the video name which is like: 'v_xxxxxxx',if different ,please have a modify.
		save_dir: the destination path's final direction name.
		step: num of frames between each two extracted frames
		bound: bi-bound parameter
	:return: no returns
	'''
	
	# video_name, save_dir, step, bound = augs
	# video_path = os.path.join(videos_root, video_name.split('_')[1], video_name)
	
	# provide two video-read methods: cv2.VideoCapture() and skvideo.io.vread(), both of which need ffmpeg support
	
	# videocapture=cv2.VideoCapture(video_path)
	# if not videocapture.isOpened():
	#     print 'Could not initialize capturing! ', video_name
	#     exit()
	try:
		videocapture = skvideo.io.vread(video_path)
	except:
		print('{} read error! '.format(video_name))
		return 0
	print(video_name)
	# if extract nothing, exit!
	if videocapture.sum() == 0:
		print('Could not initialize capturing', video_name)
		exit()
	len_frame = len(videocapture)
	frame_num = 0
	image, prev_image, gray, prev_gray = None, None, None, None
	num0 = 0
	while True:
		# frame=videocapture.read()
		if num0 >= len_frame:
			break
		frame = videocapture[num0]
		num0 += 1
		if frame_num == 0:
			image = np.zeros_like(frame)
			gray = np.zeros_like(frame)
			prev_gray = np.zeros_like(frame)
			prev_image = frame
			prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
			frame_num += 1
			# to pass the out of stepped frames
			step_t = step
			while step_t > 1:
				# frame=videocapture.read()
				num0 += 1
				step_t -= 1
			continue
		
		image = frame
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		frame_0 = prev_gray
		frame_1 = gray
		##default choose the tvl1 algorithm
		dtvl1 = cv2.createOptFlow_DualTVL1()
		flowDTVL1 = dtvl1.calc(frame_0, frame_1, None)
		save_flows(flowDTVL1, image, image_path, flow_x_path, flow_y_path, frame_num, bound)  # this is to save flows and img.
		prev_gray = gray
		prev_image = image
		frame_num += 1
		# to pass the out of stepped frames
		step_t = step
		while step_t > 1:
			# frame=videocapture.read()
			num0 += 1
			step_t -= 1



def ToImg(raw_flow,bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound
    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    return flow

def save_flows(flows,image, image_path, flow_x_path, flow_y_path, num,bound):
    '''
    To save the optical flow images and raw images
    :param flows: contains flow_x and flow_y
    :param image: raw image
    :param save_dir: save_dir name (always equal to the video id)
    :param num: the save id, which belongs one of the extracted frames
    :param bound: set the bi-bound to flow images
    :return: return 0
    '''
    #rescale to 0~255 with the bound setting
    flow_x=ToImg(flows[...,0],bound)
    flow_y=ToImg(flows[...,1],bound)
    if not os.path.exists(os.path.join(out_path)):
        os.makedirs(os.path.join(out_path))

    #save the image
    save_img=os.path.join(image_path,'img_{:05d}.jpg'.format(num))
    scipy.misc.imsave(save_img,image)

    #save the flows
    save_x=os.path.join(flow_x_path,'_{:05d}.jpg'.format(num))
    save_y=os.path.join(flow_y_path,'_{:05d}.jpg'.format(num))
    flow_x_img=Image.fromarray(flow_x)
    flow_y_img=Image.fromarray(flow_y)
    scipy.misc.imsave(save_x,flow_x_img)
    scipy.misc.imsave(save_y,flow_y_img)
    return 0

def run_optical_flow(vid_item):
	vid_path = vid_item[0]
	vid_id = vid_item[1]
	vid_name = vid_path.split('/')[-1].split('.')[0]
	out_full_path = os.path.join(out_path, vid_name)
	try:
		os.mkdir(out_full_path)
	except OSError:
		pass
	
	current = current_process()
	# dev_id = (int(current._identity[0]) - 1) % NUM_GPU
	image_path = '{}/img'.format(out_full_path)
	flow_x_path = '{}/flow_x'.format(out_full_path)
	flow_y_path = '{}/flow_y'.format(out_full_path)
	
	# cmd = os.path.join(df_path + 'build/extract_gpu')+' -f {} -x {} -y {} -i {} -b 20 -t 1 -d {} -s 1 -o {} -w {} -h {}'.format(
	#     quote(vid_path), quote(flow_x_path), quote(flow_y_path), quote(image_path), dev_id, out_format, new_size[0], new_size[1])
	
	# cmd = os.path.join(
	# 	df_path + 'build/denseFlow') + ' -f {} -x {} -y {} -i {} -b 20 -t 1 -s 1 -o {} -w {} -h {}'.format(
	# 	quote(vid_path), quote(flow_x_path), quote(flow_y_path), quote(image_path), out_format, new_size[0],
	# 	new_size[1])
	#
	# os.system(cmd)
	
	print('{} {} done'.format(vid_id, vid_name))
	sys.stdout.flush()
	return True


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="extract optical flows")
	parser.add_argument("--src_dir", type=str, default='./UCF-101',
	                    help='path to the video data')
	parser.add_argument("--out_dir", type=str, default='./ucf101_frames',
	                    help='path to store frames and optical flow')
	parser.add_argument("--df_path", type=str, default='./dense_flow/',
	                    help='path to the dense_flow toolbox')
	
	parser.add_argument("--new_width", type=int, default=0, help='resize image width')
	parser.add_argument("--new_height", type=int, default=0, help='resize image height')
	
	parser.add_argument("--num_worker", type=int, default=1)
	# parser.add_argument("--num_gpu", type=int, default=2, help='number of GPU')
	parser.add_argument("--out_format", type=str, default='dir', choices=['dir', 'zip'],
	                    help='path to the dense_flow toolbox')
	parser.add_argument("--ext", type=str, default='avi', choices=['avi', 'mp4'],
	                    help='video file extensions')
	parser.add_argument('--step', default=1, type=int, help='gap frames')
	parser.add_argument('--bound', default=15, type=int, help='set the maximum of optical flow')
	
	args = parser.parse_args()
	
	step = args.step
	bound = args.bound
	out_path = args.out_dir
	src_path = args.src_dir
	num_worker = args.num_worker
	df_path = args.df_path
	out_format = args.out_format
	ext = args.ext
	new_size = (args.new_width, args.new_height)
	# NUM_GPU = args.num_gpu
	
	if not os.path.isdir(out_path):
		print("creating folder: " + out_path)
		os.makedirs(out_path)
	
	vid_list = glob.glob(src_path + '/*/*.' + ext)
	print(len(vid_list))
	pool = Pool(num_worker)
	pool.map(run_optical_flow, zip(vid_list, xrange(len(vid_list))))
