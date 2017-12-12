#These scripts refer to "https://github.com/carpedm20/DCGAN-tensorflow" and "https://github.com/RichardYang40148/MidiNet/tree/master/v1"
from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
#import SharedArray as sa
from sklearn.utils import shuffle
from ops import *
from utils import *
import midi
from midi_manipulation import *

class MidiNet(object):
    def __init__(self, sess, is_crop=False,
                 batch_size=72, sample_size = 72, output_w=16,output_h=128,
                 y_dim=12, prev_dim=1, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=1, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None, gen_dir= None, chroma = True):
        sample_size = batch_size
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.output_w = output_w
        self.output_h = output_h
        self.chroma = chroma
        
        self.y_dim = y_dim
        self.prev_dim = prev_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn0 = batch_norm(name='d_bn0')
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')


        if self.prev_dim:
            self.g_prev_bn0 = batch_norm(name='g_prev_bn0')
            self.g_prev_bn1 = batch_norm(name='g_prev_bn1')
            self.g_prev_bn2 = batch_norm(name='g_prev_bn2')
            self.g_prev_bn3 = batch_norm(name='g_prev_bn3')


        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')
        

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            self.build_model()

    def build_model(self):
    
        self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        
        self.prev_bar = tf.placeholder(tf.float32, [self.batch_size] + [self.output_w, self.output_h, self.c_dim],
                                    name='prev_bar')
            
        self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_w, self.output_h, self.c_dim],
                                    name='real_images')
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + [self.output_w, self.output_h, self.c_dim],
                                        name='sample_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim],
                                name='z')

        self.z_sum = tf.summary.histogram("z", self.z)

        
        self.G = self.generator(self.z, self.y, self.prev_bar)
        self.D, self.D_logits, self.fm = self.discriminator(self.images, self.y, reuse=False)

        self.sampler = self.sampler(self.z, self.y, self.prev_bar)
        self.D_, self.D_logits_ , self.fm_ = self.discriminator(self.G, self.y, reuse=True)
    

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits, labels = 0.9*tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits_, labels = tf.zeros_like(self.D_)))
        self.g_loss0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_logits_, labels = tf.ones_like(self.D_)))


        #Feature Matching
        self.features_from_g = tf.reduce_mean(self.fm_, reduction_indices=(0))
        self.features_from_i = tf.reduce_mean(self.fm, reduction_indices=(0))
        self.fm_g_loss1 =tf.multiply(tf.nn.l2_loss(self.features_from_g - self.features_from_i), 0.1)

        self.mean_image_from_g = tf.reduce_mean(self.G, reduction_indices=(0))
        self.mean_image_from_i = tf.reduce_mean(self.images, reduction_indices=(0))
        self.fm_g_loss2 = tf.multiply(tf.nn.l2_loss(self.mean_image_from_g - self.mean_image_from_i), 0.01)


        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = self.g_loss0 + self.fm_g_loss1 + self.fm_g_loss2

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        
        

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
    
        if config.dataset == 'MidiNet_v1':
            # change the file path to your dataset
            data_X = np.load('./cur_copy.npy')
            if config.c_dim > 1:
                data_X = np.load('./poly_bars.npy')
            prev_X = data_X[:len(data_X)-1]
            data_X = data_X[1:]
            if config.chroma == True:
                data_y = np.load('./chromaVec.npy')[1:]
                if config.c_dim > 1:
                    data_y = np.load('./poly_chroma.npy')[1:]
                data_X, prev_X, data_y= shuffle(data_X,prev_X,data_y,random_state=0)
            else:
                data_X, prev_X = shuffle(data_X,prev_X, random_state=0)
                
            print "Data_X shape : ", data_X.shape
            print "Prev_X shape: ", prev_X.shape

            data_X = np.transpose(data_X,(0,2,3,1))
            prev_X = np.transpose(prev_X,(0,2,3,1))
        
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.global_variables_initializer().run()

        self.g_sum = tf.summary.merge([self.z_sum, self.d__sum, 
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        sample_z = np.random.normal(0, 1, size=(self.sample_size , self.z_dim))
        sample_files = data_X[0:self.sample_size]
        
        save_images(data_X[np.arange(len(data_X))[:5]]*1, [1, 5],
        './{}/Train.png'.format(config.sample_dir))
        
        
        sample_images = data_X[0:self.sample_size]
        counter = 0
        start_time = time.time()

        # if self.load(self.checkpoint_dir):
        #      print(" [*] Load SUCCESS")
        # else:
        #      print(" [!] Load failed...")
        

        sample_labels = sloppy_sample_labels()
        print 'sample_labels : ',sample_labels.shape
        for epoch in xrange(config.epoch):
            
            batch_idxs = len(data_X) // config.batch_size
            
            

            for idx in xrange(0, batch_idxs):
                
                batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
                prev_batch_images = prev_X[idx*config.batch_size:(idx+1)*config.batch_size]
                batch_labels = np.zeros((72,12))
                if config.chroma == True:
                    batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_labels = batch_labels.reshape(config.batch_size,12)
                '''
                Note that the mu and sigma are set to (-1,1) in the experiment of the paper :
                "MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation"
                However, the result are similar by using (0,1)
                '''
                batch_z = np.random.normal(0, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)

                
                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                    feed_dict={ self.images: batch_images, self.z: batch_z, self.y:batch_labels, self.prev_bar:prev_batch_images })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z, self.y:batch_labels,self.prev_bar:prev_batch_images })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                # We've tried to run more d_optim and g_optim, while getting a better result by running g_optim twice in this MidiNet version.
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z, self.y:batch_labels,self.prev_bar:prev_batch_images })
                self.writer.add_summary(summary_str, counter)

                if config.c_dim > 1:
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z, self.y:batch_labels,self.prev_bar:prev_batch_images })
                    self.writer.add_summary(summary_str, counter)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z, self.y:batch_labels,self.prev_bar:prev_batch_images })
                    self.writer.add_summary(summary_str, counter)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z, self.y:batch_labels,self.prev_bar:prev_batch_images })
                    self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.y:batch_labels,self.prev_bar:prev_batch_images })
                errD_real = self.d_loss_real.eval({self.images: batch_images,self.y:batch_labels })
                errG = self.g_loss.eval({self.images: batch_images, self.z: batch_z, self.y:batch_labels,self.prev_bar:prev_batch_images })
                
                
                

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={self.z: sample_z, self.images: sample_images, self.y:sample_labels, self.prev_bar:prev_batch_images }
                    )
                    #samples = (samples+1.)/2.
                    save_images(samples[:5,:], [1, 5],
                                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                    np.save('./{}/train_{:02d}_{:04d}'.format(config.gen_dir,  epoch, idx), samples)
            self.save(config.checkpoint_dir, counter)
            print("Epoch: [%2d] time: %4.4f, d_loss: %.8f" \
            % (epoch, 
                time.time() - start_time, (errD_fake+errD_real)))
    

    def run_poly(self, config):
        result_1 = []
        result_2 = []
        result_3 = []
        data_X = np.load('./poly_bars.npy')

        idx = np.random.randint(0,data_X.shape[0]-20)
        print 'priming from index :',idx
        start = data_X[idx]
        cur = [start]
        cur = np.transpose(cur,(0,2,3,1))
        if config.chroma == True:
            data_Y = np.load('./poly_chroma.npy')
            start_y = data_Y[idx:idx+20]
        result_1.extend(start[0])
        result_2.extend(start[1])
        result_3.extend(start[2])
        for i in range(20):
            batch_z = np.random.normal(0, 1, [config.batch_size, self.z_dim]).astype(np.float32)
            if config.chroma == True:
                temp = start_y[i][:]
                temp = temp.reshape(1,12)
                G = self.sess.run(self.G,
                            feed_dict={ self.z: batch_z, self.prev_bar:cur,self.y:temp})
            else:
                G = self.sess.run(self.G,
                        feed_dict={ self.z: batch_z, self.prev_bar:cur })
            ch1 = G[0][:,:,0]
            ch1_zeroes = [[0]*ch1.shape[1] for i in ch1]
            for i in xrange(len(ch1_zeroes)):
                ch1_zeroes[i][np.argmax(ch1[i])] = 1
            ch2 = G[0][:,:,1]
            ch2_zeroes = [[0]*ch2.shape[1] for i in ch2]
            for i in xrange(len(ch2_zeroes)):
                max_notes = ch2[i].argsort()[-3:]
                ch2_zeroes[i][max_notes[0]] = 1
                ch2_zeroes[i][max_notes[1]] = 1
                ch2_zeroes[i][max_notes[2]] = 1
            ch3 = G[0][:,:,2]
            ch3_zeroes = [[0]*ch3.shape[1] for i in ch3]
            for i in xrange(len(ch3_zeroes)):
                ch3_zeroes[i][np.argmax(ch3[i])] = 1
            result_1.extend(ch1_zeroes)
            result_2.extend(ch2_zeroes)
            result_3.extend(ch3_zeroes)
            cur = [[ch1_zeroes, ch2_zeroes, ch3_zeroes]]
            cur = np.transpose(cur,(0,2,3,1))
        result = [result_1, result_2, result_3]
        np.save('result_poly',result)

    def run_melody(self, config):
        result = []
        data_X = np.load('./cur_copy.npy')

        idx = np.random.randint(0,data_X.shape[0]-20)
        print 'priming from index :',idx
        start = data_X[idx]
        cur = [start]
        cur = np.transpose(cur,(0,2,3,1))
        if config.chroma == True:
            data_Y = np.load('./chromaVec.npy')
            start_y = data_Y[idx:idx+20]
        result.extend(start[0])

        for i in range(20):
            batch_z = np.random.normal(0, 1, [config.batch_size, self.z_dim]).astype(np.float32)
            if config.chroma == True:
                temp = start_y[i][:]
                temp = temp.reshape(1,12)
                G = self.sess.run(self.G,
                            feed_dict={ self.z: batch_z, self.prev_bar:cur,self.y:temp})
            else:
                G = self.sess.run(self.G,
                        feed_dict={ self.z: batch_z, self.prev_bar:cur })
            song = G[0][:,:,0]
            song_zeroes = [[0]*song.shape[1] for i in song]
            for i in xrange(len(song_zeroes)):
                song_zeroes[i][np.argmax(song[i])] = 1
                #song_zeroes[i][np.argmax(np.random.multinomial(1,0.99*song[i]/sum(song[i])))] = 1
            result.extend(song_zeroes)
            cur = [[song_zeroes]]
            cur = np.transpose(cur,(0,2,3,1))
        np.save('result',result)

    def run_melody_from_seed_midi(self,config):
        result = []
        pattern = midi.read_midifile(config.seed_midi)
        track1 = midi.Pattern(resolution = pattern.resolution)
        track1.append(pattern[0])
        track1.append(pattern[1])
        pr1 = np.asarray(midiToNoteStateMatrix(track1))
        track2 = midi.Pattern(resolution = pattern.resolution)
        track2.append(pattern[0])
        track2.append(pattern[2])
        pr2 = np.asarray(midiToNoteStateMatrix(track2))
        track3 = midi.Pattern(resolution = pattern.resolution)
        track3.append(pattern[0])
        track3.append(pattern[3])
        pr3 = np.asarray(midiToNoteStateMatrix(track3))
        res,res2, flag = splitIntoBarsMelody(pr1,pr2)
        if flag == True:
            print 'Sorry, try another midi file'
            return

        data_X = res
        start = data_X[0]
        cur = [[start]]
        cur = np.transpose(cur,(0,2,3,1))
        if config.chroma == True:
            data_Y = res2
            start_y = data_Y

        result.extend(start)
        print pr1.shape, pr2.shape, pr3.shape
        print len(data_X)
        print len(start_y)
        for i in range(1,len(start_y)):
            batch_z = np.random.normal(0, 1, [config.batch_size, self.z_dim]).astype(np.float32)
            if config.chroma == True:
                temp = start_y[i][:]
                temp = temp.reshape(1,12)
                G = self.sess.run(self.G,
                            feed_dict={ self.z: batch_z, self.prev_bar:cur,self.y:temp})
            else:
                G = self.sess.run(self.G,
                        feed_dict={ self.z: batch_z, self.prev_bar:cur })
            song = G[0][:,:,0]
            song_zeroes = [[0]*song.shape[1] for i in song]
            for i in xrange(len(song_zeroes)):
                song_zeroes[i][np.argmax(song[i])] = 1
                #song_zeroes[i][np.argmax(np.random.multinomial(1,0.99*song[i]/sum(song[i])))] = 1
            result.extend(song_zeroes)
            cur = [[song_zeroes]]
            cur = np.transpose(cur,(0,2,3,1))
        p1 = noteStateMatrixToMidi(result)
        song =  midi.Pattern()
        song.append(p1[0])
        t2 = midi.Pattern(resolution = pattern.resolution)
        t2.append(pattern[2])
        arr = midiToNoteStateMatrix(t2)
        print np.asarray(result).shape
        print np.asarray(arr).shape
        song.append(noteStateMatrixToMidi(arr)[0]) # attaching back the original chord progression.
        np.save('result',result)
        midi.write_midifile("result.mid", song)

    def run(self, config):
        if config.c_dim > 1:
            self.run_poly(config)
        else:
            if config.seed_midi == "":
                self.run_melody(config)
            else:
                self.run_melody_from_seed_midi(config)

    def discriminator(self, x, y=None, reuse=False):
        df_dim = 64
        dfc_dim = 1024
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if self.chroma == True:
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(x, yb)

        h0 = lrelu(conv2d(x, self.c_dim, k_h=2, k_w=128, name='d_h0_conv'))
        fm = h0
        if self.chroma == True:
            h0 = conv_cond_concat(h0, yb)

        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim, k_h=4, k_w=1, name='d_h1_conv')))
        h1 = tf.reshape(h1, [self.batch_size, -1])            
        h1 = tf.concat([h1], 1)

        h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
        h2 = tf.concat([h2], 1)

        h3 = linear(h2, 1, 'd_h3_lin')

        return tf.nn.sigmoid(h3), h3, fm

    def generator(self, z, y=None, prev_x = None):

        h0_prev = lrelu(self.g_prev_bn0(conv2d(prev_x, 16, k_h=1, k_w=128,d_h=1, d_w=2, name='g_h0_prev_conv')))
        h1_prev = lrelu(self.g_prev_bn1(conv2d(h0_prev, 16, k_h=2, k_w=1, name='g_h1_prev_conv')))
        h2_prev = lrelu(self.g_prev_bn2(conv2d(h1_prev, 16, k_h=2, k_w=1, name='g_h2_prev_conv')))
        h3_prev = lrelu(self.g_prev_bn3(conv2d(h2_prev, 16, k_h=2, k_w=1, name='g_h3_prev_conv')))

        if self.chroma == True:
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = tf.concat([z], 1)

        h0 = tf.nn.relu(self.g_bn0(linear(z, 1024, 'g_h0_lin')))
        h0 = tf.concat([h0], 1)

        h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*2*1, 'g_h1_lin')))

        h1 = tf.reshape(h1, [self.batch_size, 2, 1, self.gf_dim * 2])
        if self.chroma == True:
            h1 = conv_cond_concat(h1, yb)
        h1 = conv_prev_concat(h1, h3_prev)

        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, 4, 1, self.gf_dim * 2],k_h=2, k_w=1,d_h=2, d_w=2 ,name='g_h2')))
        if self.chroma == True:
            h2 = conv_cond_concat(h2, yb)
        h2 = conv_prev_concat(h2, h2_prev)

        h3 = tf.nn.relu(self.g_bn3(deconv2d(h2, [self.batch_size, 8, 1, self.gf_dim * 2],k_h=2, k_w=1,d_h=2, d_w=2 ,name='g_h3')))
        if self.chroma == True:
            h3 = conv_cond_concat(h3, yb)
        h3 = conv_prev_concat(h3, h1_prev)

        h4 = tf.nn.relu(self.g_bn4(deconv2d(h3, [self.batch_size, 16, 1, self.gf_dim * 2],k_h=2, k_w=1,d_h=2, d_w=2 ,name='g_h4')))
        if self.chroma == True:
            h4 = conv_cond_concat(h4, yb)
        h4 = conv_prev_concat(h4, h0_prev)

        return tf.nn.sigmoid(deconv2d(h4, [self.batch_size, 16, 128, self.c_dim],k_h=1, k_w=128,d_h=1, d_w=2, name='g_h5'))

    def sampler(self, z, y=None, prev_x=None):
        tf.get_variable_scope().reuse_variables()
        h0_prev = lrelu(self.g_prev_bn0(conv2d(prev_x, 16, k_h=1, k_w=128, d_h=1, d_w=2,name='g_h0_prev_conv')))
        h1_prev = lrelu(self.g_prev_bn1(conv2d(h0_prev, 16, k_h=2, k_w=1, name='g_h1_prev_conv')))
        h2_prev = lrelu(self.g_prev_bn2(conv2d(h1_prev, 16, k_h=2, k_w=1, name='g_h2_prev_conv')))
        h3_prev = lrelu(self.g_prev_bn3(conv2d(h2_prev, 16, k_h=2, k_w=1, name='g_h3_prev_conv')))
        

        if self.chroma == True:
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = tf.concat([z], 1)

        h0 = tf.nn.relu(self.g_bn0(linear(z, 1024, 'g_h0_lin')))
        h0 = tf.concat([h0], 1)

        h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*2*1, 'g_h1_lin')))

        h1 = tf.reshape(h1, [self.batch_size, 2, 1, self.gf_dim * 2])
        if self.chroma == True:
            h1 = conv_cond_concat(h1, yb)
        h1 = conv_prev_concat(h1, h3_prev)

        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, 4, 1, self.gf_dim * 2],k_h=2, k_w=1,d_h=2, d_w=2 ,name='g_h2')))
        if self.chroma == True:
            h2 = conv_cond_concat(h2, yb)
        h2 = conv_prev_concat(h2, h2_prev)

        h3 = tf.nn.relu(self.g_bn3(deconv2d(h2, [self.batch_size, 8, 1, self.gf_dim * 2],k_h=2, k_w=1,d_h=2, d_w=2 ,name='g_h3')))
        if self.chroma == True:
            h3 = conv_cond_concat(h3, yb)
        h3 = conv_prev_concat(h3, h1_prev)

        h4 = tf.nn.relu(self.g_bn4(deconv2d(h3, [self.batch_size, 16, 1, self.gf_dim * 2],k_h=2, k_w=1,d_h=2, d_w=2 ,name='g_h4')))
        if self.chroma == True:
            h4 = conv_cond_concat(h4, yb)
        h4 = conv_prev_concat(h4, h0_prev)

        return tf.nn.sigmoid(deconv2d(h4, [self.batch_size, 16, 128, self.c_dim],k_h=1, k_w=128,d_h=1, d_w=2, name='g_h5'))

    def save(self, checkpoint_dir, step):
        model_name = "MidiNet.model"
        model_dir = "%s_%s" % (self.dataset_name, self.output_w)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.output_w)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        print checkpoint_dir

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False