# ZSL_music_tagging

Zero-shot learning for music auto-tagging and classification.
This code works for both MSD and FMA dataset.
All pre-filtering of tracks and tags were performed beforehand. 
We provide the list of tracks/tags and their necessary matrices.


[Zero-shot Learning for Audio-based Music Classification and Tagging](https://arxiv.org/abs/1907.02670), _Jeong Choi\*, Jongpil Lee\*, Jiyoung Park, Juhan Nam_
_(\* : equally contributed authors)_ Accepted at [ISMIR 2019](https://ismir2019.ewi.tudelft.nl/?q=accepted-papers)


### Separately provided data : _[Link]()_
```
├─ data_common
  ├─ msd
    ├─ tag_key_split_TGSPP.p 
        : tag split used in paper
        # data_split_tag.py에 옵션(dataset, filename)을 주고 실행시키면 생성됨
        # tag_ids_in_key_order.p로부터 8:2 비율로 스플릿되어 [train, test]의 형태로 저장됨
        # ex) [[162,11,24,26,3,121,16,75],[124,83]] 과 같이 저장된 pickle파일
    ├─ track_keys_AB_TRSPP_TGSPP.p 
    ├─ track_keys_A_TRSPP_TGSPP.p
    ├─ track_keys_B_TRSPP_TGSPP.p
    ├─ track_keys_C_TRSPP_TGSPP.p
        : track splits used in paper
          
    ├─ all_tag_to_track_bin_matrix.p
    ├─ tag_ids_in_key_order.p
    # 순서대로 저장된 tag_id 의 1 dimension list
    ├─ track_ids_in_key_order.p
    # 순서대로 저장된 track_id의 1 dimension list
    ├─ track_id_to_file_path_dict.p
    ├─ tag_key_to_id_dict.p 
       
  ├─ fma
    ├─ ... : same as above 

       
├─ data_tag_vector
  ├─ msd
    ├─ ttr_ont_tag_1126_to_glove_dict.p
        : GloVe vector data 
          (filtered using Tagtraum genre ontology) 
  ├─ fma
    ├─ genre_id_to_inst_posneg40_cnt_norm_dict.p
    # 장르별로 사용된 악기의 갯수를 1*40 numpy array 로 표준화(평균0,표준편차1로 수렴)하여 dictionary형태로 저장
    ├─ genre_id_to_inst_posneg40_conf_norm_dict.p    
    # 장르별로 사용된 악기의 신뢰도(confidence)를 1*40 numpy array 로 표준화(평균0,표준편차1로 수렴)하여 dictionary형태로 저장
        : Instrument vector data
```


### Audio (mel-spectogram) preparation (in 'scripts' folder)

```console  
python preprocess_audio_msd.py --dir_wav PATH_TO_MSD_AUDIO_WAV --dir_mel PATH_FOR_SAVING_MEL_FILES
```


### Tag/track split data preparation (in 'scripts' folder)

 First, prepare tag splits (train / test tags)

```console  
python data_split_tag.py --dataset msd --tag_split_name TGS01 
```

 Using the tag split, prepare track splits (train / valid for AB, A, B)

```console  
python data_split_track.py --dataset msd --tag_split_name TGS01  --track_split_name TRS01 
```



### Model training / inference / evaluation
 
Training 

```console  
python train.py --dataset msd --track_split_name TRS01 --tag_split_name TGS01 --tag_vector_type glove --epochs 20 --track_split A
```

Inference 

```console  
python extract_embeddings_multi.py --load_weights PATH_TO_WEIGHT_FILE
```

Evaluation

```console  
python eval.py --load_weights PATH_TO_WEIGHT_FILE
```
