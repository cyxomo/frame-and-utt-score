import sys

def fun(finpp, trials, foutpp):

	uttlist = []
	foutlist = []
	for i in range(20):
		ii = i+1
		uttlist.append([])

	for i in range(20):
		ii = i+1
		fin = finpp + str(ii) +'.scp'
		with open(fin, 'r') as fi:
			for line in fi:
				part=line.split()
				uttlist[i].append(part[0])
		fout = foutpp + str(ii) + '.trl'
		foo = open(fout, 'w')
		foutlist.append(foo)

	with open(trials, 'r') as tr:
		for line in tr:
			part = line.split()
			for i in range(20):
				if part[1] in uttlist[i]:
					foutlist[i].write(line)
					continue

if __name__ == '__main__':
	fin = '/work1/lilt/kaldi-161111/egs/sre08/v5/exp_cnn/fisher_2000/nnet3/cnn_4_8_pi2000_po400_l6_splice10/eva_male/dvector.'
	trials = '/work1/lilt/kaldi-161111/egs/sre08/v5/data_fbank/fisher_test/male/eva_male/trials-1.trl'
	fout = '/work7/lilt/170421-dvector/ver/frame2utt/trials/male/trials_'

	fun(fin, trials, fout)