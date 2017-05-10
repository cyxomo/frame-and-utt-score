#encsssssssssssssoding=utf-8
import numpy as np
from multiprocessing import Pool
import sys


def cos_distance(x, y):
    # x = np.array(x)
    # y = np.array(y)

    return np.sum(x * y) #/ (((np.sum(x**2))**(1 / 2)) * (np.sum(y**2)**(1 / 2)))



def norm_train_data(model_file, out_file):
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
                feature_vec = np.array(feature_vec)
                modeldict[now_file_name].append(feature_vec)

                now_file_name = "[None]"
                continue
            else:
                line = line.split()
                assert(len(line) == 400)
                feature_vec = np.array([float(i) for i in line])
                feature_vec = np.array(feature_vec)
                modeldict[now_file_name].append(feature_vec)

    modelmean = {} 
    score_mean_dict = {}
    score_std_dict = {}   
    with open(out_file, 'w') as fout:
        for key in modeldict.keys():
            trainscorelist = []
            meanutt = np.mean(modeldict[key],0)
            modelmean[key] = 20 * meanutt / np.linalg.norm(meanutt, ord=2)

            for ff in modeldict[key]:
                frame_score = cos_distance(modelmean[key], ff)
                trainscorelist.append(frame_score)
            score_mean_dict[key] = np.mean(trainscorelist)
            score_std_dict[key] = np.std(trainscorelist)

            out = key+' '
            for vl in modelmean[key]:
                out = out + vl +' '
            out = out + score_mean_dict[key]+ ' '
            out = out + score_std_dict[key] + '\n'
            fout.write(out)

        del modeldict

if __name__ == '__main__':
    model_file= sys.argv[1]
    out_file = sys.argv[2]
    norm_train_data(model_file, out_file)