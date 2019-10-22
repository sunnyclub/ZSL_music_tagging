import pickle
import os
import json
import numpy as np

import sys
sys.path.append('../')

from util import fma_utils

'''
  From 163 genres, 
   we removed 6 due to their lack of instrument annotation, 
   -> leaving total of 157 genres. 

'''

'''
  Procedure 
  
  1) track by instrument (binary or confidence) vector
  2) track by genre vector
  3) genre by instrument (count or summed confidence) vector

'''



'''
  1) track x instrument vector  

'''


'''

  Open-MIC dataset
  악기로 라벨링된 오디오 데이터
'''

DATA_ROOT = '/media/iu/openmic-2018'
OPENMIC = np.load(os.path.join(DATA_ROOT, 'openmic-2018.npz'))

with open(os.path.join(DATA_ROOT, 'class-map.json'), 'r') as f:  # classmap은 "악기":int(num) 으로 이루어진 dictionary
    class_map = json.load(f)

instrument_txt_list = []
for _k in list(class_map.keys()):
    instrument_txt_list.append(_k)
instrument_txt_list.sort()

X, Y_true, Y_mask, sample_key = OPENMIC['X'], OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']
# X:(20000,10,128)
# Y_true : (20000,20)  -> 레이블에 대한 신뢰도, 여기선 0.5를 임계값으로 설정해 0,5보다 크면 왼쪽/작으면 오른쪽/
# Y_mask : (20000,20)  -> 20개의 악기 중 어떤것이 쓰였는가를 bool로 표현
#sample_key:(20000, )

CONFIDENCE_THRESHOLD = 0.5

song_key_by_inst_key_posneg_40bin_matrix = np.zeros((20000, 40))
song_key_by_inst_key_posneg_40conf_matrix = np.zeros((20000, 40))

for song_idx in range(20000):
    for inst_idx in range(20):
        if Y_mask[song_idx][inst_idx] == True:
            if Y_true[song_idx][inst_idx] > CONFIDENCE_THRESHOLD:
                song_key_by_inst_key_posneg_40bin_matrix[song_idx][inst_idx] = 1
                song_key_by_inst_key_posneg_40conf_matrix[song_idx][inst_idx] = Y_true[song_idx][inst_idx] - CONFIDENCE_THRESHOLD
            else:
                song_key_by_inst_key_posneg_40bin_matrix[song_idx][inst_idx+20] = 1
                song_key_by_inst_key_posneg_40conf_matrix[song_idx][inst_idx+20] = CONFIDENCE_THRESHOLD - Y_true[song_idx][inst_idx]



track_id_to_inst_posneg_40bin_dict = dict()
for idx in range(len(sample_key)):  # 20000개
    track_id_to_inst_posneg_40bin_dict[int(sample_key[idx].split('_')[0])] = song_key_by_inst_key_posneg_40bin_matrix[idx]
# trackID: [0,0,1,0,0,...........|20idx|0,0,0,0,1,0,0,0...,0]

track_id_to_inst_posneg_40conf_dict = dict()
for idx in range(len(sample_key)):
    track_id_to_inst_posneg_40conf_dict[int(sample_key[idx].split('_')[0])] = song_key_by_inst_key_posneg_40conf_matrix[idx]
# trackID: [0,0,0.7,0,0,...........|20idx|0,0,0,0,0.65,0,0,0...,0]

'''
  2) track x genre 
  
   : using FMA large dataset -> filter tracks with genre annotations

'''

tracks = fma_utils.load('/media/iu/fma_metadata/tracks.csv')
features = fma_utils.load('/media/iu/fma_metadata/features.csv')
echonest = fma_utils.load('/media/iu/fma_metadata/echonest.csv')

genres = fma_utils.load('/media/iu/fma_metadata/genres.csv')

np.testing.assert_array_equal(features.index, tracks.index)  # 두 어레이가 다르면 에러메시지 발생(AssertionError)
assert echonest.index.isin(tracks.index).all()  # false이면 error assertion

print(tracks.shape, features.shape, echonest.shape)
#     (106574,52)    (106574,518)     (13129,249)

track_ids = tracks.index

tracks_large = tracks['set', 'subset'] <= 'large'   # all true
track_genres_top = tracks.loc[tracks_large, ('track', 'genre_top')].values.tolist() # 106574개의 상위장르라벨,(string)
track_genres = tracks.loc[tracks_large, ('track', 'genres')].values.tolist()  #106574개의 장르라벨(idx),멀티일수도
track_ids = tracks.loc[tracks_large].index  # 106574개의 트랙id
genre_titles = genres['title'].tolist()  # 163개의 장르제목
genre_ids = genres.index.tolist() # 163개의장르id

track_ids_with_genres = []
track_id_to_genre_id_dict = {}
for i in range(len(track_genres)):
    if len(track_genres[i]) > 0:
        track_ids_with_genres.append(track_ids[i])
        track_id_to_genre_id_dict[track_ids[i]] = track_genres[i]
    else:
        continue

print('track_ids_with_genres', len(track_ids_with_genres)) # 104343

genre_ids.sort()

track_ids_with_inst = []
for _key in sample_key:
    track_ids_with_inst.append(int(_key.split('_')[0]))

print('track_ids_with_inst', len(track_ids_with_inst)) # 20000

# Here we used prefiltered tracks (19466) and genres (157)
prefiltered_track_ids_in_key_order = pickle.load(open('data_common/fma/track_ids_in_key_order.p', 'rb')) # 트랙id리스트
prefiltered_tag_ids_in_key_order = pickle.load(open('data_common/fma/tag_ids_in_key_order.p', 'rb')) # 장르id리스트

track_key_to_genre_key_binary_matrix = []

for key, t_id in enumerate(prefiltered_track_ids_in_key_order): # 19466개
    curr_binary = np.zeros(len(prefiltered_tag_ids_in_key_order)) # 157개
    for curr_genre_id in track_id_to_genre_id_dict[t_id]:  # 트랙id:장르id 로 이루어진 딕셔너리
        curr_binary[prefiltered_tag_ids_in_key_order.index(curr_genre_id)] = 1

    track_key_to_genre_key_binary_matrix.append(curr_binary)
#track_key_to_genre_key_binary_matrix
# track id    genre_one-hot
# 0 line      0 0 0 0 0 1 0 0 0
# 1           1 0 0 0 1 0 0 0 0
# 2           0 0 0 1 0 0 0 0 0
# 4           0 0 0 0 0 0 0 0 1
# 7           0 0 0 0 0 1 0 1 0
track_key_to_genre_key_binary_matrix = np.array(track_key_to_genre_key_binary_matrix)

print('track_key_to_genre_key_binary_matrix shape ', track_key_to_genre_key_binary_matrix.shape)
# (19466, 157)
genre_key_to_track_key_binary_matrix = track_key_to_genre_key_binary_matrix.T
# (157, 19466)
# 라인은 장르(태그)를 의미.. 첫번째줄이 1이다 라는것은 1이써있는 번째 트랙이 해당장르임을 나타냄
# ex)  0 1 0 0 0 0 0 0 0 1 0 0
#      1 0 0 1 1 1 0 0 1 0 0 0 <- 0, 3,4,5,8번쨰 음악이 장르1이다.




'''
  3) genre x inst
    (save as id to vector dictionary)
'''

# track_id_to_inst_posneg_40bin_dict / track_id_to_inst_posneg_40conf_dict


genre_id_to_inst_posneg40_cnt_dict = {}
genre_id_to_inst_posneg40_conf_dict = {}

for genre_key in range(len(prefiltered_tag_ids_in_key_order)): # 157개의 장르에 대해
    genre_id = prefiltered_tag_ids_in_key_order[genre_key] # 순서에 맞게 장르id 얻어내기

    genre_id_to_inst_posneg40_cnt_dict[genre_id] = np.zeros((40,))
    genre_id_to_inst_posneg40_conf_dict[genre_id] = np.zeros((40,))

    curr_track_keys = np.argwhere(genre_key_to_track_key_binary_matrix[genre_key] == 1).squeeze()
    # 장르가 genre_key인 곡의 idx를 리스트로 뽑아냄

    for _track_key in curr_track_keys:
        _track_id = prefiltered_track_ids_in_key_order[_track_key]
        _curr_track_inst_bin = track_id_to_inst_posneg_40bin_dict[_track_id] # 1*40짜리 numpy arr. 악기에 대한것
        _curr_track_inst_conf = track_id_to_inst_posneg_40conf_dict[_track_id] # 1*40짜리 numpy arr. conf에 대한것.

        genre_id_to_inst_posneg40_cnt_dict[genre_id] += _curr_track_inst_bin
        # 누적되어서 더해진다. ex: [1,0,0,1,0]+[1,1,0,0,0] = [2,1,0,1,0]
        # 결과적으로 genre_id:[10,2,0,24,0,..........12,1] 장르아이디별 사용된 악기 수의 딕셔너리
        genre_id_to_inst_posneg40_conf_dict[genre_id] += _curr_track_inst_conf
        # 마찬가지로 신뢰도합.. genre_id:[4.124,3.12,0,26.212,.....,6.32,0.7](1*40)
'''
  Standardization along genre dimension 
'''

genre_id_to_inst_posneg40_cnt_norm_dict = {}
genre_id_to_inst_posneg40_conf_norm_dict = {}

for genre_key in range(len(prefiltered_tag_ids_in_key_order)):  #157개
    genre_id = prefiltered_tag_ids_in_key_order[genre_key]

    curr_genre_vector = genre_id_to_inst_posneg40_cnt_dict[genre_id]
    _mean = curr_genre_vector.mean()  # 평균
    _std = curr_genre_vector.std()    # 표준편차: 루트 분산

    if _std == 0: #표준편차가 0이면 평평함, 사실상 가능성0
        print("Error normalizing ! (shouldn't happend since using pre-filtered tags / tracks)", genre_id, 'cnt')
        exit(0)
    curr_genre_vector_norm = (curr_genre_vector - _mean) / _std    # 표준편차가 1이 되도록 정규화시키는식
    genre_id_to_inst_posneg40_cnt_norm_dict[genre_id] = curr_genre_vector_norm
    #genre_id : [1.123 , -0.3, -0.24, .... .. , 2.13] 의 식

    curr_genre_vector = genre_id_to_inst_posneg40_conf_dict[genre_id]
    _mean = curr_genre_vector.mean()
    _std = curr_genre_vector.std()

    if _std == 0:
        print("Error normalizing ! (shouldn't happend since using pre-filtered tags / tracks)", genre_id, 'conf')
        exit(0)
    curr_genre_vector_norm = (curr_genre_vector - _mean) / _std
    genre_id_to_inst_posneg40_conf_norm_dict[genre_id] = curr_genre_vector_norm
    # 위와 같다, 이 과정을 거치면 장르별로 좀더 균일하게 되지 않나 싶다.
    # 실험결과,, 표준편차는 1로 수렴하고, 평균도 0으로 수렴한다.



savename = 'data_tag_vector/fma/genre_id_to_inst_posneg40_cnt_dict.p'
with open(savename, 'wb') as handle:
    pickle.dump(genre_id_to_inst_posneg40_cnt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(savename + ' : has been saved')

savename = 'data_tag_vector/fma/genre_id_to_inst_posneg40_conf_dict.p'
with open(savename, 'wb') as handle:
    pickle.dump(genre_id_to_inst_posneg40_conf_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(savename + ' : has been saved')

savename = 'data_tag_vector/fma/genre_id_to_inst_posneg40_cnt_norm_dict.p'
with open(savename, 'wb') as handle:
    pickle.dump(genre_id_to_inst_posneg40_cnt_norm_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(savename + ' : has been saved')

savename = 'data_tag_vector/fma/genre_id_to_inst_posneg40_conf_norm_dict.p'
with open(savename, 'wb') as handle:
    pickle.dump(genre_id_to_inst_posneg40_conf_norm_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(savename + ' : has been saved')









