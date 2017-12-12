All the pre-processing scripts are placed in the jupyter notebooks 'Melody Preprocessing' and 'Polyphonic Preprocessing'. 
These scripts can be used to convert the midi dataset to the input images for the network. 

The 'saved models' directory contains few pre-trained models. Swap the model in the checkpoint directory with any of these models to run a pre-trained model. 

Command to train the network:

<code>python main.py --is_train True</code>

Command to generate music:

<code>python main.py --batch_size 1</code>

Add addition command line argument <code>--chroma False</code> to stop the network to train using 1D chroma vector. To train
the network for multi-track music use the additional argument <code>--c_dim #num_of_tracks</code>.

Acknowledgements :<br/>
This MIDI generator is inspired from the MidiNet implementation @ https://github.com/RichardYang40148/MidiNet/tree/master/v1 <br/>
Script for converting midi to notestate matrix and the reverse were referred from https://github.com/hexahedria/biaxial-rnn-music-composition/blob/master/midi_to_statematrix.py


