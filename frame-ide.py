#encoding=utf-8
import numpy as np
from multiprocessing import Pool


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
                # 开始一个Speaker的句子
                line = line.split()
                assert(len(line) == 2)
                now_file_name = line[0]
                now_frame_id = 0
                continue

            elif ']' in line:
                # 当前Speaker的最后一行，处理完后去掉Speaker
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
                # 中间的句子
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


def acc_stat(file, topn):
    right = 0
    tot = 0
    with open(file) as f:
        for line in f:
            line = line.split()

            file = line[0]
            rank = int(line[2])

            if rank <= topn:
                right += 1
            tot += 1
    return 1.0 * right / tot


def process_mean_file(test_file, test_len_file, out_file):
    with open(out_file, 'w') as fout:
        pre_test_file_name = ""
        pre_frame_cnt = 0
        pre_sum_feature = np.zeros(400)
        for test_file_name, test_feature, frame_id in parser_test(test_file, test_len_file):
            if test_file_name != pre_test_file_name:
                if pre_test_file_name != "":
                    fout.write(" ".join([
                        pre_test_file_name,
                        "[\n" + " ".join([str(i) for i in (pre_sum_feature / pre_frame_cnt)]) + " ]",
                        ]))
                    fout.write("\n")

                    pre_test_file_name = ""
                    pre_frame_cnt = 0
                    pre_sum_feature = np.zeros(400)

            pre_test_file_name = test_file_name
            pre_frame_cnt += 1
            pre_sum_feature += test_feature


if __name__ == '__main__':

    #sample('test.ark', 'haha.txt', line_cnt=223+218+146+3)
    #sample('test.ark', 'haha.ark', line_cnt=2920)
    #process_all_distance('train.ark', 'test.ark', 'feats.len', 'distance222.txt')
    print(acc_stat('distance.txt', 1))

    #process_mean_file('test.ark', 'feats.len', 'mean_test.ark')
