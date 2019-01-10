import matplotlib.pyplot as plt
from collections import defaultdict
import os

processes = defaultdict(list)
logdir = './logdir/'
for file in os.listdir(logdir):
    print(file)
    path = logdir + file
    with open(path, 'r') as f:
        for line in f:
            if 'GABYMEMINFO' in line:
                line = line[:-1]
                infos = line[line.find('GABYMEMINFO'):]
                _, pid, mem_increase, *test_name = infos.split(' ')
                processes[file[:-8] + '_pid_' + str(pid)].append(
                    (float(mem_increase), ' '.join(test_name)))


for process_name, tests in processes.items():
    mem_usage_over_time = [0]
    for test in tests:
        mem_usage_over_time.append(mem_usage_over_time[-1] + test[0])
    mem_usage_over_time = mem_usage_over_time[1:]
    plt.plot(mem_usage_over_time)
    plt.title(process_name)
    plt.xlabel('nb of tests executed.')
    plt.ylabel('current memory usage in MB.')
    plt.savefig('./' + process_name + '.png')
    plt.show()

    test_with_num = [(i,) + test for i, test in enumerate(tests[1:])]

    test_with_num.sort(key=lambda x: x[1])

    print('\n', process_name)
    bigger_leak = test_with_num[-5:]
    bigger_leak.sort(key=lambda x: x[0])
    for leak in bigger_leak:
        print('Test number', leak[0], 'leaked', leak[1], 'MB. The name is', leak[2])
