<!--
 * @Author: XiaoHu22213 2484246430@qq.com
 * @Date: 2024-10-26 12:11:00
 * @LastEditors: XiaoHu22213 2484246430@qq.com
 * @LastEditTime: 2024-10-27 18:52:18
 * @FilePath: \undefinede:\Desktop\新建文件夹 (2)\MS-FR-Head\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# MS-FR-Head

github地址：https://github.com/XiaoHu22213/MS-FR-Head

百度网盘地址：通过网盘分享的文件：MS-FR-Head
链接: https://pan.baidu.com/s/1M97FWnI0yW9ZS4qPZ3aOIg?pwd=htvn 提取码: htvn 

<<<<<<< HEAD
权重文件地址：通过网盘分享的文件：权重文件
链接: https://pan.baidu.com/s/16mdTLP4zdulBRF0D-hEG8A?pwd=jw4v 提取码: jw4v
--来自百度网盘超级会员v1的分享

=======
>>>>>>> f18a5d2c9f1b61bc8e994793cc3663813712c82d
# 环境准备
项目基于CTR-GCN, info-gcn, FR-Head进行改进

python >=3.8
需要安装
pip install -e torchlight
pip install torchpack == 0.0.3
pip install tensorboardX

# 数据集获取

1.百度网盘获取处理完后的数据集，数据集有两种，分别位于/data文件下的原数据集 以及 /data/angle文件下的角度数据集

<<<<<<< HEAD
2.data/angle文件下有get_angle.py文件，用于获取角度数据

=======
>>>>>>> f18a5d2c9f1b61bc8e994793cc3663813712c82d
# 模型训练

模型采用双模型融合，分别为infogcn(FR_Head)和SNC-CTR-GCN两个模型。

Ⅰ .infogcn(FR_Head)模型训练方式：


1. python main_gpu.py --config ./config/uav/32frame_1.yaml --work-dir ./work/32frame_1

2. python main_gpu.py --config ./config/uav/128frame_1.yaml --work-dir ./work/128frame_1

3. python main_gpu.py --config ./config/uav/angle_FR_Head_1.yaml --work-dir ./work/angle_FR_Head_1


4. python main_gpu.py --config ./config/uav/FR_Head_1.yaml --work-dir ./work/FR_Head_1

5. python main_gpu.py --config ./config/uav/FR_Head_2.yaml --work-dir ./work/FR_Head_2

6. python main_gpu.py --config ./config/uav/FR_Head_6.yaml --work-dir ./work/FR_Head_6

7. python main_gpu.py --config ./config/uav/motion_1.yaml --work-dir ./work/motion_1

8. python main_gpu.py --config ./config/uav/motion_2.yaml --work-dir ./work/motion_2

9. python main_gpu.py --config ./config/uav/motion_6.yaml --work-dir ./work/motion_6

Ⅱ .SNC-CTR-GCN流训练方式:

1. python main_gpu.py --config config\uav\no_logit\joint.yaml

2. python main_gpu.py --config config\uav\no_logit\bone.yaml

3. python main_gpu.py --config config\uav\no_logit\joint_motion.yaml

4. python main_gpu.py --config config\uav\no_logit\bone_motion.yaml

5. python main_gpu.py --config config\uav\no_logit\joint_SNC.yaml

6. python main_gpu.py --config config\uav\no_logit\bone_SNC.yaml


# 测试新数据集

infogcn(FR_Head)

1. python main_cpu.py --phase test --config ./config/test/32frame_1.yaml --save-score true --work-dir ./weight/1 --weights work_dir/32frame_1/runs-95-12160.pt

2. python main_cpu.py --phase test --config ./config/test/128frame_1.yaml --save-score true --work-dir ./weight/1 --weights work_dir/32frame_1/runs-91-11648.pt

3. python main_cpu.py --phase test --config ./config/test/angle_FR_Head_1.yaml --save-score true --work-dir ./weight/1 --weights work_dir/32frame_1/runs-103-13184.pt

4. python main_cpu.py --phase test --config ./config/test/FR_Head_1.yaml --save-score true --work-dir ./weight/1 --weights work_dir/32frame_1/runs-105-13440.pt

5. python main_cpu.py --phase test --config ./config/test/FR_Head_2.yaml --save-score true --work-dir ./weight/1 --weights work_dir/32frame_1/runs-107-13696.pt

6. python main_cpu.py --phase test --config ./config/test/FR_Head_6.yaml --save-score true --work-dir ./weight/1 --weights work_dir/32frame_1/runs-105-13440.pt

7. python main_cpu.py --phase test --config ./config/test/motion_1.yaml --save-score true --work-dir ./weight/1 --weights work_dir/32frame_1/runs-110-14080.pt

8. python main_cpu.py --phase test --config ./config/test/motion_2.yaml --save-score true --work-dir ./weight/1 --weights work_dir/32frame_1/runs-102-13056.pt

9. python main_cpu.py --phase test --config ./config/test/motion_6.yaml --save-score true --work-dir ./weight/1 --weights work_dir/32frame_1/runs-102-13056.pt


SNC-CTR-GCN


1. python main_cpu.py --phase test --weight E:\Desktop\测试\SNC-CTR-GCN\work_dir\uav\ctrgcn_joint_v2\runs-63-16128.pt --save-score true --config config/uav/joint.yaml

2. python main_cpu.py --phase test --weight E:\Desktop\测试\SNC-CTR-GCN\work_dir\uav\ctrgcn_joint_v2\runs-63-16128.pt --save-score true --config config/uav/bone.yaml

3. python main_cpu.py --phase test --weight E:\Desktop\测试\SNC-CTR-GCN\work_dir\uav\ctrgcn_joint_v2\runs-63-16128.pt --save-score true --config config/uav/joint_motion.yaml

4. python main_cpu.py --phase test --weight E:\Desktop\测试\SNC-CTR-GCN\work_dir\uav\ctrgcn_joint_v2\runs-63-16128.pt --save-score true --config config/uav/bone_motion.yaml

5. python main_cpu.py --phase test --weight E:\Desktop\测试\SNC-CTR-GCN\work_dir\uav\ctrgcn_joint_v2\runs-63-16128.pt --save-score true --config config/uav/joint_SNC.yaml

6. python main_cpu.py --phase test --weight E:\Desktop\测试\SNC-CTR-GCN\work_dir\uav\ctrgcn_joint_v2\runs-63-16128.pt --save-score true --config config/uav/bone_SNC.yaml

# 模型融合

首先，将python main_gpu.py pharse test --weights {权重地址} --save-score true --config {选择config的地址} 测试对应的数据文件B，然后进行融合。

测试集A的融合：
python ensemble.py

测试集B的融合：
<<<<<<< HEAD
python ensemble_v2.py
=======
python ensemble_v2.py
>>>>>>> f18a5d2c9f1b61bc8e994793cc3663813712c82d
