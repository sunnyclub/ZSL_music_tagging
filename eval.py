from __future__ import print_function
import os
import numpy as np
import pickle
import argparse

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.metrics import average_precision_score
from tqdm import tqdm

def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if len(actual) == 0:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def main():
    parser = argparse.ArgumentParser(description='evaluating script')
    parser.add_argument('--eval_tag', type=int, default=1, help='0: train tags / 1: test tags')

    parser.add_argument('--load_weights', type=str,
                        default='')

    parser.add_argument('--metric', type=int, default=1,  #metric method
                        help='0:all / 1:AUC / 2:Flat hit@k and Prec@k / 3:Hierarchical Prec@k / 4:mAP / 5:Prec Recall F1')

    parser.add_argument('--k', nargs='+', type=int, default=[1,5,10],  # 1, 5, 10
                        help='')

    parser.add_argument('--eval_track_setup', type=int, default=0,   # eval_track_setup
                        help='0: B+C / 1: B / 2: C / 3: A+B+C / 4: A')

    parser.add_argument('--gzsl', type=int, default=0,
                        help='include both training and test tag set')

    args = parser.parse_args()


    if args.load_weights != None:
        exp_dir_info = args.load_weights.split('/')[1]  # exp_fma__inst-pncnt40_TGS01_TRS01_A
        args.dataset = args.load_weights.split('/')[1].split('_')[1]            # exp_[[fma]]__inst-pncnt40_TGS01_TRS01_A
        args.exp_info = args.load_weights.split('/')[1].split('_')[2]           # exp_fma_[[]]_inst-pncnt40_TGS01_TRS01_A
        args.tag_vector_type = args.load_weights.split('/')[1].split('_')[3]    # exp_fma__[[inst-pncnt40]]_TGS01_TRS01_A
        args.tag_split_name = args.load_weights.split('/')[1].split('_')[4]     # exp_fma__inst-pncnt40_[[TGS01]]_TRS01_A
        args.track_split_name = args.load_weights.split('/')[1].split('_')[5]   # exp_[[fma]]__inst-pncnt40_TGS01_[[TRS01]]_A
        args.track_split = args.load_weights.split('/')[1].split('_')[6]        # exp_[[fma]]__inst-pncnt40_TGS01_TRS01_[[A]]

        print('* dataset : ', args.dataset)          # fma
        print('* loaded exp : ', exp_dir_info)       # exp_fma__inst-pncnt40_TGS01_TRS01_A
        print('* tag_vector_type: ', args.tag_vector_type)      # inst-pncnt40
        print('* track_split : ', args.track_split)             # TGS01
        print('* track_split_name : ', args.track_split_name)   # TRS01
        print('* tag_split_name : ', args.tag_split_name)       # A

    else:
        print('Need weight path to load.')
        exit(0)

    weight_name = args.load_weights.split('/')[-1]          # w-ep_20-loss_6.29.h5
    save_path = './pred_embeddings/' + exp_dir_info + '/'   # pred_embeddings/exp_fma__inst-pncnt40_TGS01_TRS01_A
    data_common_path = os.path.join('data_common', args.dataset)        # data_common/fma
    data_tag_vector_path = os.path.join('data_tag_vector', args.dataset)    # data_tag_vector/fma

    # 1) audio embeddings
    track_key_to_embedding_dict = pickle.load(
        open(os.path.join(save_path, 'track_key_to_embedding_dict_' + weight_name + '.p'), 'rb'))

    # 2) word embeddings
    tag_key_to_embedding_dict = pickle.load(
        open(os.path.join(save_path, 'tag_key_to_embedding_dict_' + weight_name + '.p'), 'rb'))

    # 3) prepare binary matrix
    tag_key_to_track_binary_matrix = pickle.load(open(data_common_path + '/all_tag_to_track_bin_matrix.p', 'rb'))

    print('tag_key_to_track_binary_matrix', tag_key_to_track_binary_matrix.shape)  # 157*19466

    # 4) load tag split
    tag_split_file_name = data_common_path + '/tag_key_split_' + str(args.tag_split_name) + '.p'
    train_tag_keys, test_tag_keys = pickle.load(open(tag_split_file_name, 'rb'))

    # 5) load track / tag id list
    track_ids_in_order = pickle.load(open(data_common_path + '/track_ids_in_key_order.p', 'rb'))
    track_id_to_key_dict = pickle.load(open(data_common_path + '/track_id_to_key_dict.p', 'rb'))
    print('track_ids_in_order', len( track_ids_in_order))

    if args.eval_tag == 0:  # train set
        print('training tags:', len(train_tag_keys))
        curr_tag_keys = train_tag_keys
    elif args.eval_tag == 1:  # test set
        print('test tags:', len(test_tag_keys))
        curr_tag_keys = test_tag_keys

    if args.gzsl == 1:  # what is GZSL ?? : generalized zero shot learning
        print('GZSL setting')
        curr_tag_keys = train_tag_keys + test_tag_keys

    curr_track_keys = []
    for t_id in track_ids_in_order:
        curr_track_keys.append(track_id_to_key_dict[t_id])
    curr_track_keys.sort()

    curr_tag_key_to_track_binary_matrix = tag_key_to_track_binary_matrix[:, curr_track_keys]
    #

    # 5) load tag embeddings for test set
    curr_tag_embeddings = []
    tag_audio_label_table = []

    for tag_key in curr_tag_keys:
        curr_tag_embeddings.append(tag_key_to_embedding_dict[tag_key].squeeze())  # 100차원벡터들의 리스트
        tag_audio_label_table.append(curr_tag_key_to_track_binary_matrix[tag_key]) # 태그(여기선 32) * 트랙 의 bin matrix-->tag로 인덱싱했으므로 19466길이의 bin 벡터
    curr_tag_embeddings = np.array(curr_tag_embeddings)
    tag_audio_label_table = np.array(tag_audio_label_table)
    print('curr split true binary label table shape :', tag_audio_label_table.shape)   # 32 * 19466
    print('curr split tag embeddings shape :', curr_tag_embeddings.shape)   # 32 * 100차원 벡터


    # 6) load all audio embeddings
    audio_embeddings = []
    for key in range(len(track_ids_in_order)):
        audio_embeddings.append(track_key_to_embedding_dict[key])
    audio_embeddings = np.array(audio_embeddings)
    print('all audio embeddings shape :', audio_embeddings.shape)   #  19466 * 100


    # 7) prepare matrix to evaluate

    # (1) prepare prediction table
    tag_audio_pred_table = cosine_similarity(curr_tag_embeddings, audio_embeddings) # 32*100의 curr_tag_embeddings와 19466*100의 audio_embeddings의 유사도
    # 32*19466의 cos_similarity vector가 나옴!
    print('--> prediction (cosine similarity) table shape :', tag_audio_pred_table.shape)
    # (2) Filter tracks to evaluate

    # FOR B+C test set
    if args.eval_track_setup == 0:
        print('B+C test set')

        print('loading B + C track splits..')
        savename = data_common_path + '/track_keys_B_' + args.track_split_name + '_' + args.tag_split_name + '.p'
        B_track_keys_train, B_track_keys_valid = pickle.load(open(savename, 'rb'))
        filtered_track_keys_B = B_track_keys_train +  B_track_keys_valid

        savename = data_common_path + '/track_keys_C_' + args.track_split_name + '_' + args.tag_split_name + '.p'
        filtered_track_keys_C = pickle.load(open(savename, 'rb'))

        filtered_track_keys_BC = filtered_track_keys_B + filtered_track_keys_C

        filtered_tag_audio_label_table_t = tag_audio_label_table.T[filtered_track_keys_BC]  # 트랙(B+C) * 장르(여기선 32)의 bin matrix
        filtered_tag_audio_pred_table_t = tag_audio_pred_table.T[filtered_track_keys_BC]   # 트랙(B+C) * 장르(여기선32)의 cos 유사도벡터

        filtered_tag_audio_label_table_t = np.array(filtered_tag_audio_label_table_t)
        filtered_tag_audio_pred_table_t = np.array(filtered_tag_audio_pred_table_t)

        tag_audio_label_table = filtered_tag_audio_label_table_t.T
        tag_audio_pred_table = filtered_tag_audio_pred_table_t.T

        print('* FILTERED true binary label table shape :', tag_audio_label_table.shape)
        print('* FILTERED prediction table shape :', tag_audio_pred_table.shape)


    # FOR B test set
    elif args.eval_track_setup == 1:
        print('B test set')
        savename = data_common_path + '/track_keys_B_' + args.track_split_name + '_' + args.tag_split_name + '.p'
        B_track_keys_train, B_track_keys_valid = pickle.load(open(savename, 'rb'))    # 9:1
        filtered_track_keys_B = B_track_keys_train + B_track_keys_valid

        filtered_tag_audio_label_table_t = tag_audio_label_table.T[filtered_track_keys_B]
        filtered_tag_audio_pred_table_t = tag_audio_pred_table.T[filtered_track_keys_B]

        filtered_tag_audio_label_table_t = np.array(filtered_tag_audio_label_table_t)
        filtered_tag_audio_pred_table_t = np.array(filtered_tag_audio_pred_table_t)

        tag_audio_label_table = filtered_tag_audio_label_table_t.T
        tag_audio_pred_table = filtered_tag_audio_pred_table_t.T

        print('* FILTERED true binary label table shape :', tag_audio_label_table.shape)
        print('* FILTERED prediction table shape :', tag_audio_pred_table.shape)

    # FOR C test set
    elif args.eval_track_setup == 2:
        print('C test set')
        savename = data_common_path + '/track_keys_C_' + args.track_split_name + '_' + args.tag_split_name + '.p'
        filtered_track_keys_C = pickle.load(open(savename, 'rb'))

        filtered_tag_audio_label_table_t = tag_audio_label_table.T[filtered_track_keys_C, :]
        filtered_tag_audio_pred_table_t = tag_audio_pred_table.T[filtered_track_keys_C, :]

        filtered_tag_audio_label_table_t = np.array(filtered_tag_audio_label_table_t)
        filtered_tag_audio_pred_table_t = np.array(filtered_tag_audio_pred_table_t)

        tag_audio_label_table = filtered_tag_audio_label_table_t.T
        tag_audio_pred_table = filtered_tag_audio_pred_table_t.T

        print('* FILTERED true binary label table shape :', tag_audio_label_table.shape)
        print('* FILTERED prediction table shape :', tag_audio_pred_table.shape)

    # FOR A+B+C test set
    elif args.eval_track_setup == 3:
        print('*** A+B+C test ***')

        savename = data_common_path + '/track_keys_A_' + args.track_split_name + '_' + args.tag_split_name + '.p'
        A_track_keys_train, A_track_keys_valid = pickle.load(open(savename, 'rb'))
        filtered_track_keys_A= A_track_keys_train + A_track_keys_valid

        savename = data_common_path + '/track_keys_B_' + args.track_split_name + '_' + args.tag_split_name + '.p'
        B_track_keys_train, B_track_keys_valid = pickle.load(open(savename, 'rb'))
        filtered_track_keys_B = B_track_keys_train + B_track_keys_valid

        savename = data_common_path + '/track_keys_C_' + args.track_split_name + '_' + args.tag_split_name + '.p'
        filtered_track_keys_C = pickle.load(open(savename, 'rb'))

        print(" A: ", len(filtered_track_keys_A))
        print(" B: ", len(filtered_track_keys_B))
        print(" C: ", len(filtered_track_keys_C))

        filtered_track_keys = filtered_track_keys_A + filtered_track_keys_B + filtered_track_keys_C

        filtered_tag_audio_label_table_t = tag_audio_label_table.T[filtered_track_keys, :]
        filtered_tag_audio_pred_table_t = tag_audio_pred_table.T[filtered_track_keys, :]

        filtered_tag_audio_label_table_t = np.array(filtered_tag_audio_label_table_t)
        filtered_tag_audio_pred_table_t = np.array(filtered_tag_audio_pred_table_t)

        tag_audio_label_table = filtered_tag_audio_label_table_t.T
        tag_audio_pred_table = filtered_tag_audio_pred_table_t.T

        print('* FILTERED true binary label table shape :', tag_audio_label_table.shape)
        print('* FILTERED prediction table shape :', tag_audio_pred_table.shape)

    # FOR A test set
    if args.eval_track_setup == 4:
        print('*** A test ***')

        savename = 'data_matrix/track_keys_A_' + args.track_split_name + '_' + args.tag_split_name + '.p'
        A_track_keys_train, A_track_keys_valid = pickle.load(open(savename, 'rb'))
        filtered_track_keys_A = A_track_keys_train + A_track_keys_valid

        filtered_track_keys = []
        filtered_track_keys.extend(filtered_track_keys_A)

        filtered_tag_audio_label_table_t = tag_audio_label_table.T[filtered_track_keys, :]
        filtered_tag_audio_pred_table_t = tag_audio_pred_table.T[filtered_track_keys, :]

        filtered_tag_audio_label_table_t = np.array(filtered_tag_audio_label_table_t)
        filtered_tag_audio_pred_table_t = np.array(filtered_tag_audio_pred_table_t)

        tag_audio_label_table = filtered_tag_audio_label_table_t.T
        tag_audio_pred_table = filtered_tag_audio_pred_table_t.T

        print('* FILTERED true binary label table shape :', tag_audio_label_table.shape)
        print('* FILTERED prediction table shape :', tag_audio_pred_table.shape)


######이상 테스트셋 정하는 부분입니다.
# test set에 따라 tag_audio_label_table : tag * track갯수 에 맞는 shape. bin matrix
#                 tag_audio_pred_table : tag * track갯수 만큼의 코사인유사도 테이블.
#                 row_idx의 tag embedding vector(100d)와 col_idx의 track embedding vector(100d)의 cos유사도
    ## (1) AUC for Retrieval / Annotation
    # 누적분포, 그렇게 많이 쓰는 건 아니다.
    if args.metric == 0 or args.metric == 1 or args.metric == 10:  #0: all 1: auc  10:?

        auc_ret = []
        n_error = 0
        for idx in tqdm(range(tag_audio_label_table.shape[0])):   # tag갯수(unseen일 경우 32)
            try:
                auc_ret.append(roc_auc_score(tag_audio_label_table[idx, :], tag_audio_pred_table[idx, :]))
                # track갯수만큼의 길이를 가진 해당 태그의 binary vector와 유사도 track갯수만큼의길이를 가진 해당 태그의 유사도 vector의 roc_auc_score
                #                 ex) 1번태그:[0 0 0 0 1 0 0 0 0 ... 0 0 1] , 1번태그:[-0.12 0 0 0 0.82 0 0 0 0 ... 0 0 0.93]
                # cos 유사도 : -1이면 반대, 0이면 관계가 없음, 1이면 유사.
            except Exception as error:
                # print(error)
                auc_ret.append(0)
                # print('error in ', idx)
                n_error += 1

        print('Retrieval AUC :', sum(auc_ret)/(len(auc_ret) - n_error))   # 평균 AUC: 평균적인 AUC스코어 수치
        print('n error : ', n_error)

        auc_anno = []
        n_error = 0
        for idx in tqdm(range(tag_audio_label_table.shape[1])):  #track갯수(B+C)일 경우 9800
            try:
                auc_anno.append(roc_auc_score(tag_audio_label_table[:, idx], tag_audio_pred_table[:, idx]))
                # track 아이디와 태그와의관계를 조회해봄
                #            0번트랙                 0번트랙
                # 0번장르       [0]                    [0.1]
                # 1번장르       [0]                    [-0.1]
                # 2번장르       [0]                    [-0.2]
                # 3번장르       [1]                    [0.98]
                # 4번장르       [0]                    [0.1]
                # 5번장르       [0]                    [-0.06]
                # 6번장르       [1]                    [0.86]

            except Exception as error:
                n_error += 1

        print('Annotation AUC :', sum(auc_anno) / len(auc_anno))
        print('n error : ', n_error)


    ## (2) Flat hit @ k and Precision @ K for Retrieval / Annotation
    if args.metric == 0 or args.metric == 2 or args.metric == 10:
        _k_list = args.k   # [1,5,10]

        flathit_list_ret = [[] for _ in range(len(_k_list))]  # [[],[],[]]
        mapk_list_ret = [[] for _ in range(len(_k_list))]  #  [[],[],[]]

        for idx in tqdm(range(tag_audio_pred_table.shape[0])):  # tag갯수

            for k_idx in range(len(_k_list)): # [1,5,10]
                _k = _k_list[k_idx]
                curr_full_pred = tag_audio_pred_table[idx, :]   # tag와 트랙과의 유사도, 길이 9800의 cos유사도벡터
                curr_pred_indices = curr_full_pred.argsort()[-_k:][::-1]  # 가장 높은거 몇개를 뽑아내는거같은데?

                curr_full_label = tag_audio_label_table[idx, :]   # tag로 조회되는 track길이의 binary matrix.
                curr_label_indices = np.argwhere(curr_full_label==1)  # 1이 위치한 곳
                if len(curr_label_indices) > 1:
                    curr_label_indices = curr_label_indices.squeeze()

                _hit = 0
                for _pred_idx in curr_pred_indices:
                    if _pred_idx in curr_label_indices:
                        _hit += 1

                # _hit = _hit / _k
                if _hit > 0:
                    _hit = 1
                flathit_list_ret[k_idx].append(_hit)

                curr_mapk = apk(curr_label_indices, curr_pred_indices, k=_k)
                mapk_list_ret[k_idx].append(curr_mapk)

        for k_idx in range(len(_k_list)):
            print('Retrieval Flat hit@k (k=' +  str(_k_list[k_idx]) + ') :', sum(flathit_list_ret[k_idx]) / len(flathit_list_ret[k_idx]))
            print('Retrieval average precision@k (k=' + str(_k_list[k_idx]) + ') :', sum(mapk_list_ret[k_idx]) / len(mapk_list_ret[k_idx]))

        flathit_list_anno = [[] for _ in range(len(_k_list))]
        mapk_list_anno = [[] for _ in range(len(_k_list))]

        for idx in tqdm(range(tag_audio_pred_table.shape[1])):
            for k_idx in range(len(_k_list)):
                _k = _k_list[k_idx]
                curr_full_pred = tag_audio_pred_table[:, idx]
                curr_pred_indices = curr_full_pred.argsort()[-_k:][::-1]

                curr_full_label = tag_audio_label_table[:, idx]
                curr_label_indices = np.argwhere(curr_full_label == 1)
                if len(curr_label_indices) > 1:
                    curr_label_indices = curr_label_indices.squeeze()

                _hit = 0
                for _pred_idx in curr_pred_indices:
                    if _pred_idx in curr_label_indices:
                        _hit += 1
                if _hit > 0:
                    _hit = 1
                flathit_list_anno[k_idx].append(_hit)

                curr_mapk = apk(curr_label_indices, curr_pred_indices, k=_k)
                mapk_list_anno[k_idx].append(curr_mapk)

        for k_idx in range(len(_k_list)):
            print('Annotation Flat hit@k (k=' + str(_k_list[k_idx]) + ') :', sum(flathit_list_anno[k_idx]) / len(flathit_list_anno[k_idx]))
            print('Annotation average precision@k (k=' + str(_k_list[k_idx]) + ') :', sum(mapk_list_anno[k_idx]) / len(mapk_list_anno[k_idx]))


    ## (3) Hierarchical Precision hit @ k for Retrieval / Annotation





    ## (4) mAP for Retrieval / Annotation
    # 추천시스템에서도 쓰임 MAP-7 : 7개를 추천해줬는데 나머지 몇개를 맞췄냐. 1등만 추천하고 나머지 다틀려도 100점
    # 노래 음악추천을 해줄 때 내가 제일 좋아하는거 하나만 맞춰도 100점.
    # 1등을 틀리고 2등~7등까지 다맞추면 몇점.

    if args.metric == 0 or args.metric == 4 or args.metric == 10:

        map_list_ret = []

        for idx in tqdm(range(tag_audio_pred_table.shape[0])):
            curr_full_pred = tag_audio_pred_table[idx, :]
            curr_full_label = tag_audio_label_table[idx, :]

            curr_ap = average_precision_score(curr_full_label, curr_full_pred)

            map_list_ret.append(curr_ap)

        print('Retrieval mAP :', sum(map_list_ret) / len(map_list_ret))

        map_list_anno = []

        for idx in tqdm(range(tag_audio_pred_table.shape[1])):
            curr_full_pred = tag_audio_pred_table[:, idx]
            curr_full_label = tag_audio_label_table[:, idx]

            curr_ap = average_precision_score(curr_full_label, curr_full_pred)

            map_list_anno.append(curr_ap)

        print('Annotation mAP :', sum(map_list_anno) / len(map_list_anno))



    ## (5) Precision, Recall, F-Score for Retrieval / Annotation

    if args.metric == 0 or args.metric == 5:

        thresholds = list(np.arange(0.0, 1.0, 0.10))
        # thresholds = []

        f1_list_ret = []
        precision_list_ret = []
        recall_list_ret = []
        for threshold in thresholds:
            for idx in range(tag_audio_label_table.shape[0]):
                curr_pred = tag_audio_pred_table[idx, :]
                curr_pred_bin = [1 if curr_pred[i] >= threshold else 0 for i in range(curr_pred.shape[0])]
                curr_label_bin = tag_audio_label_table[idx, :]
                # print('curr pred idx:', np.where(curr_pred_bin))
                # print('curr label idx:', np.where(curr_label_bin))

                curr_p, curr_r, curr_f1, support = precision_recall_fscore_support(curr_label_bin, curr_pred_bin,
                                                                                   average='binary')

                f1_list_ret.append(curr_f1)
                precision_list_ret.append(curr_p)
                recall_list_ret.append(curr_r)

            print('threshold :', threshold, ' f1 retrieval :', sum(f1_list_ret)/len(f1_list_ret),
                  ' precision :', sum(precision_list_ret) / len(precision_list_ret),
                  ' recall :', sum(recall_list_ret) / len(recall_list_ret))


        f1_list_anno = []
        precision_list_anno = []
        recall_list_anno = []
        for threshold in thresholds:
            for idx in range(tag_audio_label_table.shape[1]):
                curr_pred = tag_audio_pred_table[:, idx]
                curr_pred_bin = [1 if curr_pred[i] >= threshold else 0 for i in range(curr_pred.shape[0])]
                curr_label_bin = tag_audio_label_table[:, idx]
                # print('curr pred idx:', np.where(curr_pred_bin))
                # print('curr label idx:', np.where(curr_label_bin))

                curr_p, curr_r, curr_f1, support = precision_recall_fscore_support(curr_label_bin, curr_pred_bin,
                                                                                   average='binary')
                f1_list_anno.append(curr_f1)
                precision_list_anno.append(curr_p)
                recall_list_anno.append(curr_r)

            print('threshold :', threshold, ' f1 annotation :', sum(f1_list_anno) / len(f1_list_anno),
                  ' precision :', sum(precision_list_anno) / len(precision_list_anno),
                  ' recall :', sum(recall_list_anno) / len(recall_list_anno))

if __name__ == '__main__':
    main()

