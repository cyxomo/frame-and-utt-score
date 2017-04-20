#encsssssssssssssoding=utf-8
import numpy as np
from multiprocessing import Pool
import sys


def cos_distance(x, y):
    # x = np.array(x)
    # y = np.array(y)

    return np.sum(x * y) #/ (((np.sum(x**2))**(1 / 2)) * (np.sum(y**2)**(1 / 2)))


def norm_feature(x):
    x = np.array(x)
    #return x
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


def utt2utt_score(model_file, test_file, trials_file, out_score):
    model = {}
    for speaker, feature in parser_train(model_file):
        if speaker in model:
            print(speaker, 'has in model!')
            assert(speaker not in model)

        model[speaker] = norm_feature(feature)

    testutt = {}
    for uttintest, feature in parser_train(test_file):
        if uttintest in testutt:
            print(uttintest, 'has in test.ark!')
            assert(uttintest not in testutt)

        testutt[uttintest] = norm_feature(feature)

    fout = open(out_score, 'w')
    with open(trials_file, 'r') as trials:
        for line in trials:
            part = line.split()
            score = cos_distance(model[part[0]], testutt[part[1]])
            out = part[0] + ' ' +part[1] +' ' +str(score) + ' ' + part[2] + '\n'
            fout.write(out)


def frame2utt_score(model_file, test_len_file, test_file, trials_file, out_file):

    model = {}
    for speaker, feature in parser_train(model_file):
        if speaker in model:
            print(speaker, 'has in model!')
            assert(speaker not in model)

        model[speaker] = norm_feature(feature)
    all_speaker = model.keys()
    testdict = {}
    testlist = []
    with open(out_file, 'w') as fout:
        # fout.write("[{}]\n".format(",".join(all_speaker)))
        for test_file_name, test_feature, frame_id in parser_test(test_file, test_len_file):
            test_feature = norm_feature(test_feature)
            if not test_file_name in testlist:
                testlist.append(test_file_name)
                testdict[test_file_name] = []
            testdict[test_file_name].append(test_feature)

        with open(trials_file, 'r') as trials:
            for line in trials:
                part = line.split()
                scorelist = []
                # part[0] = spk  part[1] = utt
                for testframevec in testdict[part[1]]:
                    frame_score = cos_distance(model[part[0]], testframevec)
                    scorelist.append(frame_score)
                utt_score = np.mean(scorelist)
                out = part[0] + ' ' +part[1] +' ' +str(utt_score) + ' ' + part[2] + '\n'
                fout.write(out)

def frame2frame_score(model_file, test_len_file, test_file, trials_file, out_file):
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
    for key in modeldict.keys():
        meanutt = np.mean(modeldict[key],0)
        modelmean[key] = meanutt / np.linalg.norm(meanutt, ord=2)
    del modeldict

    testdict = {}
    testlist = []
    for test_file_name, test_feature, frame_id in parser_test(test_file, test_len_file):
        test_feature = norm_feature(test_feature)
        if not test_file_name in testlist:
            testlist.append(test_file_name)
            testdict[test_file_name] = []
        testdict[test_file_name].append(test_feature)

    with open(out_file, 'w') as fout:
        with open(trials_file, 'r') as trials:
            for line in trials:
                part = line.split()
                scorelist = []
                # part[0] = spk  part[1] = utt
                for testframevec in testdict[part[1]]:
                    frame_score = cos_distance(modelmean[part[0]], testframevec)
                    scorelist.append(frame_score)
                score_mean = np.mean(scorelist)
                score_std = np.std(scorelist)
                scoframe = []
                print scorelist
                for ss in scorelist:
                    scoframe .append( (ss - score_mean) / score_std )

                finscore = np.mean(scoframe)

                out = part[0] + ' ' +part[1] +' ' +str(finscore) + ' ' + part[2] + '\n'
                fout.write(out)


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



if __name__ == '__main__':
    model_file= sys.argv[1]
    test_file=sys.argv[2]
    test_len_file=sys.argv[3]
    trials_file = sys.argv[4]
    out_file = sys.argv[5]
    utt2utt_score(model_file, test_file, trials_file, out_file)
    #frame2utt_score(model_file, test_len_file, test_file, trials_file, out_file)
    #frame2frame_score(model_file, test_len_file, test_file, trials_file, out_file)