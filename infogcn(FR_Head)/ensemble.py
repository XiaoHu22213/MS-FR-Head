import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    r1 = "./work_dir/32frame_1"
    r2 = "./work_dir/128frame_1"
    r3 = "./work_dir/angle_FR_Head_1"
    r4 = "./work_dir/FR_Head_1"
    r5 = "./work_dir/FR_Head_2"
    r6 = "./work_dir/FR_Head_6"
    r7 = "./work_dir/motion_1"
    r8 = "./work_dir/motion_2"
    r9 = "./work_dir/motion_6"


    with open('../data/test_label_A.npy', 'rb') as f:
        label = np.load(f)

    with open(os.path.join(r1, 'best_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(r2, 'best_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    
    with open(os.path.join(r3, 'best_score.pkl'), 'rb') as r3:
        r3 = list(pickle.load(r3).items())

    with open(os.path.join(r4, 'best_score.pkl'), 'rb') as r4:
        r4 = list(pickle.load(r4).items())

    with open(os.path.join(r5, 'best_score.pkl'), 'rb') as r5:
        r5 = list(pickle.load(r5).items())

    with open(os.path.join(r6, 'best_score.pkl'), 'rb') as r6:
        r6 = list(pickle.load(r6).items())

    with open(os.path.join(r7, 'best_score.pkl'), 'rb') as r7:
        r7 = list(pickle.load(r7).items())

    with open(os.path.join(r8, 'best_score.pkl'), 'rb') as r8:
        r8 = list(pickle.load(r8).items())

    with open(os.path.join(r9, 'best_score.pkl'), 'rb') as r9:
        r9 = list(pickle.load(r9).items())
    

    right_num = total_num = right_num_5 = 0
    best = 0.0

    total_num = 0
    right_num = 0
    #alpha = [0.7,0.7,0.7,0.6,0.4,0.4]
    alpha = [0.5, 0.5, 1, 1.2, 1.2, 1.4, 0.3, 0.3, 0.3]
    #alpha = [1,1,1,1,1,1,1,1,1]
    #alpha = [0.9, 0.9, 0.7, 0.3, 0.4, 0.1]
    #alpha = alpha / np.sum(alpha)

    # 创建一个列表来存储融合结果
    fused_results = []

    for i in tqdm(range(len(label))):
        l = label[i]
        _, r11 = r1[i]
        _, r22 = r2[i]
        _, r33 = r3[i]
        _, r44 = r4[i]
        _, r55 = r5[i]
        _, r66 = r6[i]
        _, r77 = r7[i]
        _, r88 = r8[i]
        _, r99 = r9[i]

        r = r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2] + r44 * alpha[3] + r55 * alpha[4] + r66 * alpha[5] + r77 * alpha[6] + r88 * alpha[7] + r99 * alpha[8]

        # 将融合结果添加到列表中
        fused_results.append(r)

        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r_max = np.argmax(r)
        right_num += int(r_max == int(l))
        total_num += 1

    # 将融合结果列表转换为numpy数组
    fused_results_array = np.array(fused_results)

    # 保存融合结果为prey.npy文件
    np.save('pred.npy', fused_results_array)

    acc = right_num / total_num
    print(acc, alpha)
    if acc > best:
        best = acc
        best_alpha = alpha
    acc5 = right_num_5 / total_num

    print(best, best_alpha)
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
    print('Fusion results saved to prey.npy')
