#encoding=utf-8
import numpy as np
from multiprocessing import Pool
import sys

def cos_distance(x, y):
    # x = np.array(x)
    # y = np.array(y)

    return np.sum(x * y) #/ (((np.sum(x**2))**(1 / 2)) * (np.sum(y**2)**(1 / 2)))


def norm_feature(x):
    x = np.array(x)
    return x / ((np.sum(x**2.0))**(1.0 / 2))


def sample(file, out_file, line_cnt):
    with open(file) as f:
        with open(out_file, 'w') as fout:
            cnt = 0
            for a in f:
                fout.write(a)

                cnt += 1
                if cnt == line_cnt:
                    break


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


def process_all_distance(model_file, test_file, test_len_file, out_file):
    def cal_distance(model, test_feature, test_speaker):
        all_dis = []
        for speaker in all_speaker:
            feature = model[speaker]
            all_dis.append([speaker, cos_distance(feature, test_feature)])
        sorted_dis = sorted(all_dis, key=lambda x: -x[1])

        rank = -1
        for i in range(len(sorted_dis)):
            if sorted_dis[i][0] == test_speaker:
                rank = i + 1
                break

        return rank, sorted_dis

    model = {}
    for speaker, feature in parser_train(model_file):
        if speaker in model:
            print(speaker, 'has in model!')
            assert(speaker not in model)

        model[speaker] = norm_feature(feature) # cos距离是就不用Norm了
    all_speaker = model.keys()

    with open(out_file, 'w') as fout:
        # fout.write("[{}]\n".format(",".join(all_speaker)))
        for test_file_name, test_feature, frame_id in parser_test(test_file, test_len_file):
            test_feature = norm_feature(test_feature) # cos距离是就不用Norm了

            test_speaker = test_file_name.split('-')[0]
            rank, sorted_dis = cal_distance(model, test_feature, test_speaker)
            # fout.write("{} {} {}\n".format(test_file_name, frame_id, rank))
            fout.write(" ".join([
                test_file_name,
                str(frame_id),
                str(rank),
                str(sorted_dis[rank - 1][0]),
                str(round(sorted_dis[rank - 1][1], 4)),
                str(sorted_dis[0][0]),
                str(round(sorted_dis[0][1], 4)),
            ]))
            fout.write("\n")
            # fout.write("[{},{},{},[{}]]\n".format(test_speaker, frame_id, rank, ",".join([str(i[1]) for i in sorted_dis])))

def utt2utt_score(model_file, test_file, out_score):
    def cal_distance(model, test_feature, test_speaker):
        all_dis = []
        for speaker in all_speaker:
            feature = model[speaker]
            all_dis.append([speaker, cos_distance(feature, test_feature)])
        sorted_dis = sorted(all_dis, key=lambda x: -x[1])

        rank = -1
        for i in range(len(sorted_dis)):
            if sorted_dis[i][0] == test_speaker:
                rank = i + 1
                break

        return rank, sorted_dis
    model = {}
    for speaker, feature in parser_train(model_file):
        if speaker in model:
            print(speaker, 'has in model!')
            assert(speaker not in model)

        model[speaker] = norm_feature(feature)
    all_speaker = model.keys()

    with open(out_file, 'w') as fout:
        # fout.write("[{}]\n".format(",".join(all_speaker)))
        for test_file_name, test_feature in parser_train(test_file):
            test_feature = norm_feature(test_feature) # cos距离是就不用Norm了

            test_speaker = test_file_name.split('-')[0]
            rank, sorted_dis = cal_distance(model, test_feature, test_speaker)
            # fout.write("{} {} {}\n".format(test_file_name, frame_id, rank))
            fout.write(" ".join([
                str(sorted_dis[rank - 1][0]),
                test_file_name,
                str(rank),
                str(round(sorted_dis[rank - 1][1], 4)),
            ]))
            fout.write("\n")


def frame2utt_score(model_file, test_file, test_len_file, out_file):
    def cal_distance(model, test_feature, test_speaker):
        all_dis = []
        for speaker in all_speaker:
            feature = model[speaker]
            all_dis.append([speaker, cos_distance(feature, test_feature)])
        sorted_dis = sorted(all_dis, key=lambda x: -x[1])

        rank = -1
        for i in range(len(sorted_dis)):
            if sorted_dis[i][0] == test_speaker:
                rank = i + 1
                break

        return rank, sorted_dis

    model = {}
    for speaker, feature in parser_train(model_file):
        if speaker in model:
            print(speaker, 'has in model!')
            assert(speaker not in model)

        model[speaker] = norm_feature(feature) 
    all_speaker = model.keys()

    with open(out_file, 'w') as fout:
        # fout.write("[{}]\n".format(",".join(all_speaker)))
        for test_file_name, test_feature, frame_id in parser_test(test_file, test_len_file):
            test_feature = norm_feature(test_feature) 

            test_speaker = test_file_name.split('-')[0]
            rank, sorted_dis = cal_distance(model, test_feature, test_speaker)
            # fout.write("{} {} {}\n".format(test_file_name, frame_id, rank))
            fout.write(" ".join([
                str(sorted_dis[rank - 1][0]),
                test_file_name,
                str(rank),
                str(round(sorted_dis[rank - 1][1], 4)),
            ]))
            fout.write("\n")
            # fout.write("[{},{},{},[{}]]\n".format(test_speaker, frame_id, rank, ",".join([str(i[1]) for i in sorted_dis])))

def frame2frame_score(model_file, test_file, test_len_file, out_file):
    def cal_distance2(model, test_feature, test_speaker):
        all_dis = []
        for speaker in all_speaker:
            feature = model[speaker]
            all_dis.append([cos_distance(feature, test_feature)])

        return all_dis

    modeldict = {}
    with open(model_file, 'r') as mf:
        now_file_name = "[None]"
        for line in mf:
            if len(line.strip()) == 0:
                continue
            if '[' in line:
                line = line.split()
                assert(len(line) == 2)
                now_file_name = line[0]
                modeldict[now_file_name] = []
                continue
            elif ']' in line:
                line = line.split()
                assert(len(line) == 401)
                assert(']' in line)

                line = line[:-1]

                assert(len(line) == 400)
                assert(']' not in line)

                feature_vec = np.array([float(i) for i in line])
                feature_vec = norm_feature(feature_vec)
                modeldict[now_file_name].append(feature_vec)

                now_file_name = "[None]"
                continue
            else:
                line = line.split()
                assert(len(line) == 400)
                feature_vec = np.array([float(i) for i in line])
                feature_vec = norm_feature(feature_vec)
                modeldict[now_file_name].append(feature_vec)
    modelmean = {} 
    score_mean_dict = {}
    score_std_dict = {}   
    for key in modeldict.keys():
        trainscorelist = []
        meanutt = np.mean(modeldict[key],0)
        modelmean[key] = meanutt / np.linalg.norm(meanutt, ord=2)
        for ff in modeldict[key]:
            frame_score = cos_distance(modelmean[key], ff)
            trainscorelist.append(frame_score)
        score_mean_dict[key] = np.mean(trainscorelist)
        score_std_dict[key] = np.std(trainscorelist)
    del modeldict
    all_speaker = modelmean.keys()

    testdict = {}
    testlist = []
    for test_file_name, test_feature, frame_id in parser_test(test_file, test_len_file):
        test_feature = norm_feature(test_feature)
        if not test_file_name in testlist:
            testlist.append(test_file_name)
            testdict[test_file_name] = []
        testdict[test_file_name].append(test_feature)

    with open(out_file, 'w') as fout:
        # fout.write("[{}]\n".format(",".join(all_speaker)))
        for utttest in testdict.key():
            score_mat = []
            test_speaker = utttest.split('-')[0]
            for frame_vec in testdict[utttest]:
                all_dis = cal_distance2(modelmean, frame_vec, test_speaker)
                score_mat.append(all_dis)
            score_mat = np.array(score_mat)
            score_mat = np.transpose(score_mat)
            score_mean = score_mean_dict[part[0]]
            score_std = score_std_dict[part[0]]
            spk_score_list = []
            for i in range(len(score_mat)):
                sss = np.mean( (score_mat[i] - score_mean) / score_std)
                spk_score_list.append([all_speaker[i], sss])

            sorted_dis = sorted(spk_score_list, key=lambda x: -x[1])
            rank = -1
            for i in range(len(sorted_dis)):
                if sorted_dis[i][0] == test_speaker:
                    rank = i + 1
                    break
            # fout.write("{} {} {}\n".format(test_file_name, frame_id, rank))
            fout.write(" ".join([
                str(sorted_dis[rank - 1][0]),
                test_file_name,
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
    #utt2utt_score(model_file, test_file, out_file)
    frame2utt_score(model_file, test_file, test_len_file, out_file)
    #frame2frame_score(model_file, test_file, test_len_file, out_file)
