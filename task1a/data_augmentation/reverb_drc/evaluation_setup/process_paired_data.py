import numpy as np

fr = open('fold1_train.csv').readlines()
source_save_path = 'fold1_train_source_paired.csv'
target_save_path = 'fold1_train_target_paired.csv'

title = fr[0]
fr = fr[1:]
f = fr.copy()

device_names = []

for i, elem in enumerate(f):
    f[i] = f[i].split('/')[-1].split('.')[0]
    device_names.append(f[i].split('-')[-1])


device_list = np.unique(device_names)

source_idxs = [i for i in range(len(f)) if device_names[i] == 'a']
target_idxs = [i for i in range(len(f)) if device_names[i] != 'a']

source_names = [f[i][:-2] for i in source_idxs]

target_to_source = []
for i in range(len(target_idxs)):
    if f[target_idxs[i]][-2] == '-':
        cur_target_name = f[target_idxs[i]][:-2]
    elif f[target_idxs[i]][-3] == '-':
        cur_target_name = f[target_idxs[i]][:-3]

    if cur_target_name not in source_names:
        print('not in source')
        exit()

    target_to_source.append(source_names.index(cur_target_name))

    

fws = open(source_save_path, 'w')
fwt = open(target_save_path, 'w')

fws.write(title)
fwt.write(title)

for i in range(len(target_idxs)):
    fwt.write(fr[target_idxs[i]])
    fws.write(fr[source_idxs[target_to_source[i]]])

fws.close()
fwt.close()




