import os

folder_path_result = r"/mnt/sdb/weiqiushi/Data/ApuCPathKPI202456Nx/TrainDataset/PredictTrainDataset/EvaluatePatch/PrecisionRecall"

path_precision = os.path.join(folder_path_result, "evaluatePrecision.txt")
path_recall = os.path.join(folder_path_result, "evaluateRecall.txt")
path_dice = os.path.join(folder_path_result, "evaluateDice.txt")
path_img_num = os.path.join(folder_path_result, "evaluatePrecisionRecallImageNum.txt")
path_dist = os.path.join(folder_path_result, "evaluateMaskDist.txt")

path_output_merge = os.path.join(folder_path_result, "Collect.txt")
path_output_patch_dist = os.path.join(folder_path_result, "Dist.txt")

if not os.path.exists(os.path.dirname(path_output_merge)):
    os.makedirs(os.path.dirname(path_output_merge))

with open(path_img_num, 'r') as f:
    buffer = f.readlines()

list_img_num = []
line_num = 0
for line in buffer:
    if line_num == 0:
        pass
    else:
        list_line = line.strip().split('\t')
        list_img_num.append([list_line[1], int(list_line[2])])
    line_num += 1

line_num = 0
list_precision = []
with open(path_precision, 'r') as f:
    buffer = f.readlines()
for line in buffer:
    list_line = line[:-1].split('\t')
    if line_num == 0:
        for item in list_line:
            list_precision.append([item, [], 0])
    else:
        for i, item in enumerate(list_line):
            if len(item) == 0:
                pass
            else:
                list_precision[i][2] += 1
                list_precision[i][1].append(float(item))
    line_num += 1

line_num = 0
list_recall = []
with open(path_recall, 'r') as f:
    buffer = f.readlines()
for line in buffer:
    list_line = line[:-1].split('\t')
    if line_num == 0:
        for item in list_line:
            list_recall.append([item, [], 0])
    else:
        for i, item in enumerate(list_line):
            if len(item) == 0:
                pass
            else:
                list_recall[i][2] += 1
                list_recall[i][1].append(float(item))
    line_num += 1

line_num = 0
list_dice = []
with open(path_dice, 'r') as f:
    buffer = f.readlines()
for line in buffer:
    list_line = line[:-1].split('\t')
    if line_num == 0:
        for item in list_line:
            list_dice.append([item, [], 0])
    else:
        for i, item in enumerate(list_line):
            if len(item) == 0:
                pass
            else:
                list_dice[i][2] += 1
                list_dice[i][1].append(float(item))
    line_num += 1

line_num = 0
list_dist = []
with open(path_dist, 'r') as f:
    buffer = f.readlines()
for line in buffer:
    list_line = line[:-1].split('\t')
    if line_num == 0:
        for item in list_line:
            list_dist.append([item, [], 0])
    else:
        for i, item in enumerate(list_line):
            if len(item) == 0:
                pass
            else:
                list_dist[i][2] += 1
                list_dist[i][1].append(float(item))
    line_num += 1

with open(path_output_merge, 'w') as f:
    f.write('idx\twsi name\timage num\tprecision\trecall\tdice\n')
    for idx, img_info in enumerate(list_img_num):
        f.write(str(idx + 1))
        f.write('\t')
        f.write(img_info[0])
        f.write('\t')
        f.write(str(img_info[1]))
        f.write('\t')
        for j in range(len(list_precision)):
            if list_precision[j][0] == img_info[0]:
                f.write("%.6f" % (sum(list_precision[j][1]) / list_precision[j][2]))
                f.write('\t')
                break

        for j in range(len(list_recall)):
            if list_recall[j][0] == img_info[0]:
                f.write("%.6f" % (sum(list_recall[j][1]) / list_recall[j][2]))
                f.write('\t')
                break

        for j in range(len(list_dice)):
            if list_dice[j][0] == img_info[0]:
                f.write("%.6f" % (sum(list_dice[j][1]) / list_dice[j][2]))
                f.write('\t')
                break
        f.write('\n')

with open(path_output_patch_dist, 'w') as f:
    f.write('idx\twsi name\tdist\tprecision\trecall\tdice\n')
    count = 0
    for image_idx, image_num_info in enumerate(list_img_num):
        for dist_idx, dist_info in enumerate(list_dist):
            if dist_info[0] == image_num_info[0]:
                f.write("%d"%(count + 1))
                f.write('\t')
                f.write(image_num_info[0])
                f.write('\t')
                for i in range(image_num_info[1]):
                    if i > 0:
                        f.write("%d" % (count + 1))
                        f.write('\t')
                        f.write('')
                        f.write('\t')
                    f.write('%.6f'%list_dist[image_idx][1][i])
                    f.write('\t')
                    f.write('%.6f'%list_precision[image_idx][1][i])
                    f.write('\t')
                    f.write('%.6f'%list_recall[image_idx][1][i])
                    f.write('\t')
                    f.write('%.6f'%list_dice[image_idx][1][i])
                    f.write('\t')
                    f.write('\n')
                    count += 1

                break
a = 1