
import numpy as np

rate_to_train = 0.6
trainning_data_name = 'processed_data_20_movies.txt'
onehot_dimension = 300

dataset = []
with open(trainning_data_name, 'r', encoding='utf-8') as f:
    for line in f:
        dataset.append(line)
f.close()
num_total = len(dataset)
idx = np.arange(num_total)
np.random.shuffle(idx)
new_dataset = [dataset[i] for i in idx]
train_set = new_dataset[:int(num_total*rate_to_train)]
test_set = new_dataset[int(num_total*rate_to_train):]

with open('train_set_0.6.txt', 'a', newline='\n', encoding='utf-8') as f:
    for data in train_set:
        f.write(data)
f.close()

with open('test_set_0.6.txt', 'a', newline='\n', encoding='utf-8') as f:
    for data in test_set:
        f.write(data)
f.close()
