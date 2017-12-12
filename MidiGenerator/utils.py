#These scripts refer to "https://github.com/carpedm20/DCGAN-tensorflow" and "https://github.com/RichardYang40148/MidiNet/tree/master/v1"
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
import midi
import pretty_midi
from midi_manipulation import *
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc", 
                        "sy": 1, "sx": 1, 
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv", 
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

#a sloppy way of creating sample labels, which we generate 3 samples for 24 chord types each.
def sloppy_sample_labels():
    
    sl_a = np.append(np.arange(24),np.arange(24))
    sl_a = np.append(sl_a,np.arange(24))
    sl_b = np.zeros((72, 24))
    #sl_b = np.zeros((36, 24))
    #sl_b[np.arange(72), sl_a] = 1
    sl_b[np.arange(72), sl_a] = 1
    sl_b_ch13=np.zeros((1,12))
    next_ch13=np.zeros((1,12))
    for i in range(0,len(sl_b)):
        next_ch13=np.zeros((1,12))
        if sl_b[i].argmax(axis=0) >= 12:
            next_ch13[0][sl_b[i].argmax(axis=0)-12] = 1
            next_ch13[0][11] = 1
            sl_b_ch13 = np.append(sl_b_ch13,next_ch13,axis=0)
        if sl_b[i].argmax(axis=0)  < 12:
            next_ch13[0][sl_b[i].argmax(axis=0)] = 1
            sl_b_ch13 = np.append(sl_b_ch13,next_ch13,axis=0)
    sl_b_ch13 = sl_b_ch13[1:]
    return sl_b_ch13

def generation_test(sess, dcgan, config, option, prev_bar):
  if option == 0:
    

    sample_labels = sloppy_sample_labels()
    prev_batch_images = np.zeros((72, 16, 128, 1))
    z_sample = np.random.normal(0, 1, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y:sample_labels, dcgan.prev_bar:prev_batch_images})
    return samples

  if option == 1:
    

    sample_labels = sloppy_sample_labels()
    prev_batch_images = np.tile(prev_bar,(72,1,1,1))
    z_sample = np.random.normal(0, 1, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y:sample_labels, dcgan.prev_bar:prev_batch_images})

    return samples

# split piano rolls corresponding to melody, chord progression and bass tracks into bars of size (128x16)
def splitIntoBars(pr1, pr2, pr3):
    ret = []
    ret2 = []
    flag = False
    (r1,c1) = pr1.shape
    (r2,c2) = pr2.shape
    (r3,c3) = pr3.shape
    sizes = [r1, r2, r3]
    rows = min(sizes)
    if r1 != r2 or r1 != r3:
        pr1 = pr1[:rows, :]
        pr2 = pr2[:rows, :]
        pr3 = pr3[:rows, :]
    i = 1
    while rows >= i*16:
        midi.write_midifile("chroma_temp.mid", noteStateMatrixToMidi(pr2[(i)*16-16 : (i)*16,:]))
        midi_data = pretty_midi.PrettyMIDI('chroma_temp.mid')
        try:
            chroma = midi_data.get_chroma(fs=1.0/midi_data.get_end_time())
            ret.append([pr1[(i)*16-16 : (i)*16,:], pr2[(i)*16-16 : (i)*16,:], pr3[(i)*16-16 : (i)*16,:]])
            ret2.append(chroma)
        except ZeroDivisionError:
            flag = True
        i += 1
    return np.asarray(ret), np.asarray(ret2), flag

# split piano rolls corresponding to melody, chord progression and bass tracks into bars of size (128x16)
def splitIntoBarsMelody(pr1, pr2):
    ret = []
    ret2 = []
    flag = False
    (r1,c1) = pr1.shape
    (r2,c2) = pr2.shape
    sizes = [r1, r2]
    rows = min(sizes)
    if r1 != r2:
        pr1 = pr1[:rows, :]
        pr2 = pr2[:rows, :]
    i = 1
    while rows >= i*16:
        midi.write_midifile("chroma_temp.mid", noteStateMatrixToMidi(pr2[(i)*16-16 : (i)*16,:]))
        midi_data = pretty_midi.PrettyMIDI('chroma_temp.mid')
        try:
            chroma = midi_data.get_chroma(fs=1.0/midi_data.get_end_time())
            ret.append(pr1[(i)*16-16 : (i)*16,:])
            ret2.append(chroma)
        except ZeroDivisionError:
            flag = True
        i += 1
    return np.asarray(ret), np.asarray(ret2), flag

  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        