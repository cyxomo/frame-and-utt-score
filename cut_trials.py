import sys

def fun(finpp, trials, foutpp):
	uttlist = []
	for i in range(20):
		ii = i+1
		uttlist.append([])
		fin = finpp + str(ii) +'.scp'
	with open(fin, 'r') as fi:
		for line in fi:
			part=line.split()
			uttlist.append(part[0])
	with open(fout, 'w') as fo:
		with open(trials, 'r') as tr:
			for line in tr:
				part = line.split()
				if part[1] in uttlist:
					fo.write(line)

if __name__ == '__main__':
	for nn in range(10)
	fin = '/work1/lilt/kaldi-161111/egs/sre08/v5/exp_cnn/fisher_2000/nnet3/cnn_4_8_pi2000_po400_l6_splice10/eva_female/dvector.'+ str(num)+'.scp'
	trials = '/work1/lilt/kaldi-161111/egs/sre08/v5/data_fbank/fisher_test/female/eva_female/trials-1.trl'
	fout = '/work7/lilt/170421-dvector/ver/frame2utt/trials/female/trials_1.trl'

	fun(fin, trials, fout)