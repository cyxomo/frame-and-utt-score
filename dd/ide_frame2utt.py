#encoding=utf-8
import numpy as np
from multiprocessing import Pool
import sys

def cos_distance(x, y):
    # x = np.array(x)
    # y = np.array(y)

    return np.sum(x * y) #/ (((np.sum(x**2))**(1 / 2)) * (np.sum(y**2)**(1 / 2)))


def parser_train(file):
    with open(file) as f:
        for line in f:
            line = line.split()
            assert(line[1] == '[')
            assert(line[-1] == ']')
            assert(len(line) == 400 + 3)

            speaker_id = line[0]
            feature_vec = line[2:-1]
            assert(len(feature_vec) == 400)

            feature_vec = np.array([float(i) for i in feature_vec])
            yield speaker_id, feature_vec


def parser_test(feature_file, feature_length_file):

    frame_cnt = {}
    with open(feature_length_file) as f:
        for line in f:
            file, cnt = line.split()
            frame_cnt[file] = int(cnt)

    with open(feature_file) as f:
        now_file_name = "[None]"
        for line in f:
            if len(line.strip()) == 0:
                continue

            if '[' in line:
                line = line.split()
                assert(len(line) == 2)
                now_file_name = line[0]
                now_frame_id = 0
                continue

            elif ']' in line:
                line = line.split()
                assert(len(line) == 401)
                assert(']' in line)

                line = line[:-1]

                assert(len(line) == 400)
                assert(']' not in line)

                feature_vec = np.array([float(i) for i in line])

                yield(now_file_name, feature_vec, now_frame_id)
                now_frame_id += 1

                if now_frame_id != frame_cnt[now_file_name]:
                    print('Parser ERROR: {} should have {} frame, parsed {} frame'.format(
                        now_file_name, now_frame_id, frame_cnt[now_file_name]
                    )
                    )

                now_file_name = "[None]"
                now_frame_id = -100
                continue
            else:
                line = line.split()
                assert(len(line) == 400)

                feature_vec = np.array([float(i) for i in line])
                yield(now_file_name, feature_vec, now_frame_id)
                now_frame_id += 1


def frame2utt_score(model_file, test_file, test_len_file, out_file):
    #compute cosin distance
    def cal_distance2(model, test_feature, test_speaker):
        all_dis = []
        for speaker in all_speaker:
            feature = model[speaker]
            all_dis.append(cos_distance(feature, test_feature))

        return all_dis

    # input train data
    model = {}
    for speaker, feature in parser_train(model_file):
        if speaker in model:
            print(speaker, 'has in model!')
            assert(speaker not in model)

        model[speaker] = np.array(feature) 
    all_speaker = model.keys()

    testdict = {}
    testlist = []
    for test_file_name, test_feature, frame_id in parser_test(test_file, test_len_file):
        test_feature = np.array(test_feature)
        if not test_file_name in testlist:
            testlist.append(test_file_name)
            testdict[test_file_name] = []
        testdict[test_file_name].append(test_feature)

    with open(out_file, 'w') as fout:
        # fout.write("[{}]\n".format(",".join(all_speaker)))
        for utttest in testdict.keys():
            score_mat = []
            test_speaker = utttest.split('-')[0]
            for frame_vec in testdict[utttest]:
                all_dis = cal_distance2(model, frame_vec, test_speaker)
                score_mat.append(all_dis)
            score_mat = np.array(score_mat)
            score_mat = np.transpose(score_mat)
            
            spk_score_list = []
            for i in range(len(score_mat)):
                sss = np.mean( score_mat[i] )
                spk_score_list.append([all_speaker[i], sss])
            sorted_dis = sorted(spk_score_list, key=lambda x: -x[1])
            rank = -1
            for i in range(len(sorted_dis)):
                if sorted_dis[i][0] == test_speaker:
                    rank = i + 1
                    break
            fout.write(" ".join([
                str(sorted_dis[rank - 1][0]),
                utttest,
                str(rank),
                str(round(sorted_dis[rank - 1][1], 4)),
            ]))
            fout.write("\n")
            # fout.write("[{},{},{},[{}]]\n".format(test_speaker, frame_id, rank, ",".join([str(i[1]) for i in sorted_dis])))


if __name__ == '__main__':
    model_file= sys.argv[1]
    test_file=sys.argv[2]
    test_len_file=sys.argv[3]
    out_file = sys.argv[4]
    frame2utt_score(model_file, test_file, test_len_file, out_file)