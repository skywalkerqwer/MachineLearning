# 机器学习笔记大纲
## day01
- 01_scale.py
  - 数据预处理:均值移除(标准化)
  - 让样本矩阵中的每一列的平均值为0，标准差为1
  - A = sp.scale(array)
  - array矩阵需要符合样本要求(一行一样本,一列一特征)
- 02_mms.py
  - 范围缩放
  - 样本范围在[0,1]之间
  - 手动实现范围缩放
- 03_norm.py
  - 归一化(正则化)
  - 保留样本比例,使样本每列之和为1
- 04_bin.py
  - 二值化
  - 手动实现二值化
- 05_one_hot.py
  - 独热矩阵
  - 获取码表以对不同数据使用同一个码表进行独热编码计算
  