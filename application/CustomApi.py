import numpy as np
import librosa
from librosa import display
import os

sample_rate = 22050
melBin = 128
hop = 5122
window = 1024
fft_size = 1024


def print_mel(input_name):
    y, _ = librosa.load(input_name, sr=sample_rate, mono=True)
    if y.shape[0] < 3000:
        print("audio length is too short!")
        return
    S = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=melBin)
    plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(S, ref=np.max)
    display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sample_rate, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()


def convert_to_input_mel(input_name):
    audio, _ = librosa.load(input_name, sr=sample_rate, mono=True)
    if audio.shape[0] < 3000:
        print("audio length is too short!")
        return
    D = librosa.stft(audio, n_fft=fft_size, win_length=window, window='hann', hop_length=hop)
    y = np.abs(D)

    mel_basis = librosa.filters.mel(sample_rate, fft_size, n_mels=melBin)
    y = np.dot(mel_basis, y)
    y = np.log10(1 + 10 * y)
    y = y.T
    return y


input_name = '016011.mp3'

from matplotlib import pyplot as plt

print_mel(input_name)
y = convert_to_input_mel(input_name)
from model import model_siamese_a2w_1fc
from keras.models import Model

from keras.layers import Conv1D, MaxPool1D, GlobalAvgPool1D, Dense, Activation, Input, dot, concatenate
from keras.regularizers import l2


def myModel(argDict):
    num_frame = argDict['num_frame']
    num_negative_sampling = argDict['num_negative_sampling']
    # audio anchor
    myModel.audio_input = Input(shape=(num_frame, 128), name='am_input')
    # positive word
    myModel.pos_item = Input(shape=(argDict['tag_vec_dim'],))
    # negative word
    myModel.neg_items = [Input(shape=(argDict['tag_vec_dim'],)) for j in range(num_negative_sampling)]

    # audio model

    conv1 = Conv1D(128, 4, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform',
                   name='am_conv_1')
    activ1 = Activation('relu')
    MP1 = MaxPool1D(pool_size=4)
    conv2 = Conv1D(128, 4, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform',
                   name='am_conv_2')
    activ2 = Activation('relu')
    MP2 = MaxPool1D(pool_size=4)
    conv3 = Conv1D(128, 4, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform',
                   name='am_conv_3')
    activ3 = Activation('relu')
    MP3 = MaxPool1D(pool_size=4)
    conv4 = Conv1D(128, 2, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform',
                   name='am_conv_4')
    activ4 = Activation('relu')
    MP4 = MaxPool1D(pool_size=2)
    conv5 = Conv1D(100, 1, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_uniform',
                   name='am_conv_5')

    # (batch, steps, features) -> (batch, features)
    GP = GlobalAvgPool1D()

    fc1 = Dense(100)

    # Audio Anchor
    anchor_conv1 = conv1(myModel.audio_input)
    anchor_activ1 = activ1(anchor_conv1)
    anchor_MP1 = MP1(anchor_activ1)
    anchor_conv2 = conv2(anchor_MP1)
    anchor_activ2 = activ2(anchor_conv2)
    anchor_MP2 = MP2(anchor_activ2)
    anchor_conv3 = conv3(anchor_MP2)
    anchor_activ3 = activ3(anchor_conv3)
    anchor_MP3 = MP3(anchor_activ3)
    anchor_conv4 = conv4(anchor_MP3)
    anchor_activ4 = activ4(anchor_conv4)
    anchor_MP4 = MP4(anchor_activ4)
    anchor_conv5 = conv5(anchor_MP4)
    myModel.anchor_output = GP(anchor_conv5)

    # positive word item
    myModel.pos_item_output = fc1(myModel.pos_item)

    # negative word item
    myModel.neg_item_output = [fc1(neg_item) for neg_item in myModel.neg_items]

    RQD_p = dot([myModel.anchor_output, myModel.pos_item_output], axes=1, normalize=True)
    RQD_ns = [dot([myModel.anchor_output, neg_item], axes=1, normalize=True) for neg_item in myModel.neg_item_output]

    prob = concatenate([RQD_p] + RQD_ns)

    output = Activation('linear')(prob)

    model = Model(inputs=[myModel.audio_input, myModel.pos_item] + myModel.neg_items, outputs=output)

    return model


argDict = dict()

# parser = argparse.ArgumentParser(description="load weights for predict")
argDict['load_weights'] = "weights/exp_fma_inst_pncnt40_TGS03_TRS03_A/w-ep_70-loss_3.30.h5"

# Path settings
argDict['dir_mel'] = "./fma_large_mel"
argDict['num_part'] = 12
argDict['num_negative_sampling'] = 1

argDict['global_mel_mean'] = 0.2262
argDict['global_mel_std'] = 0.2579
argDict['num_frame'] = 130

exp_dir_info = argDict['load_weights'].split('/')[1]

argDict['dataset'] = argDict['load_weights'].split('/')[1].split('_')[1]
argDict['exp_info'] = argDict['load_weights'].split('/')[1].split('_')[2]
argDict['tag_vector_type'] = argDict['load_weights'].split('/')[1].split('_')[3]
argDict['tag_split_name'] = argDict['load_weights'].split('/')[1].split('_')[4]
argDict['track_split_name'] = argDict['load_weights'].split('/')[1].split('_')[5]
argDict['track_split'] = argDict['load_weights'].split('/')[1].split('_')[6]

data_common_path = os.path.join('data_common', argDict['dataset'])
data_tag_vector_path = os.path.join('data_tag_vector', argDict['dataset'])
argDict['data_common_path'] = data_common_path
argDict['tag_vec_dim'] = 40

model = myModel(argDict)
