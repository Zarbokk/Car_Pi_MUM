import sys
import bisect
import tempfile
import random
import time

fin = open("/home/tim/Documents/Car_Pi_MUM/python_control/projectWork/training_NN/train_data/augmented_data/complete_for_training/augmented_data_FULL.csv")

s = time.clock()

if len(sys.argv) > 2:
    N = int(sys.argv[2])
else:
    N = 0
    sys.stderr.write("No number of lines provided, counting lines in input\n")
    for line in fin:
        N += 1
    fin.seek(0)

sys.stderr.write("Shuffling {0} lines...\n".format(N))
time.sleep(100)


# random permutation
def random_permutation(N):
    l = list(range(N))
    for i, n in enumerate(l):
        r = random.randint(0, i)
        l[i] = l[r]
        l[r] = n
    return l

p = random_permutation(N)
ridx = [0] * N
files = []
mx = []


#print(p)
#print(ridx)

sys.stderr.write("Computing list of temporary files\n")
for i, n in enumerate(p):
    # print(i)
    # print(n)
    pos = bisect.bisect_left(mx, n) - 1
    if pos == -1:
        files.insert(0, [n])
        mx.insert(0, n)
    else:
        files[pos].append(n)
        mx[pos] = n
#print(files)
#print(len(files))
#time.sleep(100)
P = len(files)
sys.stderr.write("Caching to {0} temporary files\n".format(P))
fps = [tempfile.TemporaryFile(mode="w+") for i in range(P)]

for file_index, line_list in enumerate(files):
    for line in line_list:
        ridx[line] = file_index

# write to each temporal file
for i, line in enumerate(fin):
    fps[ridx[i]].write(line)

for f in fps:
    f.seek(0)


sys.stderr.write("Writing to the shuffled file\n")
# write to the final shuffled file
for i in range(N):
    print(fps[ridx[p[i]]].readline().strip())

e = time.clock()

sys.stderr.write("Shuffling took an overall of {0} secs\n".format(e-s))