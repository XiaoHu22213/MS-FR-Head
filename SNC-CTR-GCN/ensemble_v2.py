import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA', 'uav-v1', 'uav-v2'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-SNC-dir', default=None)
    parser.add_argument('--bone-SNC-dir', default=None)
    parser.add_argument('--motion-dir', default=None)
    parser.add_argument('--motion-SNC-dir', default=None)

    arg = parser.parse_args()

    dataset = arg.dataset
    if 'UCLA' in arg.dataset:
        label = []
        with open('./data/' + 'NW-UCLA/' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('/mnt/netdisk/wulh/MAMP-main/data/ntu/NTU60_XSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in arg.dataset:
            npz_data = np.load('/mnt/netdisk/wulh/MAMP-main/data/ntu/NTU60_XView.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'uav' in arg.dataset:
        if 'v1' in arg.dataset:
            with open('../data/uav/v1/test_label.pkl', 'rb') as f:
                _, label = pickle.load(f)
        elif 'v2' in arg.dataset:
            with open('../data/test_label_A.npy', 'rb') as f:
                label = np.load(f)
    else:
        raise NotImplementedError

    with open(os.path.join(arg.joint_dir, 'best_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(arg.bone_dir, 'best_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    if arg.joint_SNC_dir is not None:
        with open(os.path.join(arg.joint_SNC_dir, 'best_score.pkl'), 'rb') as r3:
            r3 = list(pickle.load(r3).items())
    if arg.bone_SNC_dir is not None:
        with open(os.path.join(arg.bone_SNC_dir, 'best_score.pkl'), 'rb') as r4:
            r4 = list(pickle.load(r4).items())
    if arg.motion_dir is not None:
        with open(os.path.join(arg.motion_dir, 'best_score.pkl'), 'rb') as r5:
            r5 = list(pickle.load(r5).items())
    if arg.motion_SNC_dir is not None:
        with open(os.path.join(arg.motion_SNC_dir, 'best_score.pkl'), 'rb') as r6:
            r6 = list(pickle.load(r6).items())

    right_num = total_num = right_num_5 = 0
    best = 0.0
    if arg.joint_dir is not None and arg.bone_dir is not None:
        total_num = 0
        right_num = 0
        arg.alpha = [1, 0.9, 0.7, 0.3, 0.4, 0.1]
        #arg.alpha = [0.9, 0.9, 0.7, 0.3, 0.4, 0.1]
        #arg.alpha = arg.alpha / np.sum(arg.alpha)

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

            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3] + r55 * arg.alpha[4] + r66 * arg.alpha[5]

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
        print(acc, arg.alpha)
        if acc > best:
            best = acc
            best_alpha = arg.alpha
        acc5 = right_num_5 / total_num

    print(best, best_alpha)
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
    print('Fusion results saved to prey.npy')
