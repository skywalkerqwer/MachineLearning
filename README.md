# 机器学习笔记大纲
## day01
- 01_scale.py
  - 数据预处理:均值移除(标准化)
  - 一个样本的不同特征值差异较大时使用
  - 让样本矩阵中的每一列的平均值为0，标准差为1
  - A = sp.scale(array)
  - array矩阵需要符合样本要求(一行一样本,一列一特征)
- 02_mms.py
  - 范围缩放
  - 统一各列特征值的范围在[0,1]之间
  - 手动实现范围缩放
- 03_norm.py
  - 归一化(正则化)
  - 不关注样本特征值而更关注特征值占比时
  - 保留样本比例,使样本每列之和为1
- 04_bin.py
  - 二值化
  - 不需要详细数据,只需要关注特征值与阈值的关系时
  - 手动实现二值化
- 05_one_hot.py
  - 独热矩阵
  - 对离散的数据使用
  - 获取码表以对不同数据使用同一个码表进行独热编码计算
- 06_lbe.py
  - 对字符串使用标签编码
  - 预测完毕后进行逆向编码
- 07_loss.py
  - 绘制loss函数图像
- 08_lr.py
  - 偏导函数根据梯度下降算法求最小值
  - 设置lrate学习率使每次迭代后的点更趋近最小值点
## day02 
- 01_lr.py
  - 绘制每次梯度下降w0 w1 loss的变化曲线
  - 基于三维曲面绘制梯度下降过程中每一个点
- 02_linearRegession.py
  - 线性回归
  - 采集数据
  - 训练模型
  - 模型预测
- 03_metrics.py
  - 训练结果误差评估
  - 平均绝对值误差
  - 平均平方误差
  - 中位绝对值误差
  - R2得分
- 04_dump.py | 05_load.py
  - 模型的保存和加载
- 06_ridge.py
  - 岭回归
  - 调整正则项超参数
  - 减少异常数据对训练结果产生的影响
- 07_poly.py
  - 多项式回归模型预测
  - 不能预测超出训练范围
- 08_tree.py
  - 决策树
  - 使每次划分的子表中该特征的值全部相同
- 09_se.py
  - 正向激励
  - 构建一棵带有权重的决策树,对预测不准的样本提高权重
  - 多次重复构建不同权重的若干决策树
## day03
- 01_fi.py
  - 特征重要性
  - 代表每个特征对模型的重要性
- 02_rf.py
  - 自助聚合
      - 有放回的随机抽取部分样本构建决策树
      - 削弱某些强势样本对模型的影响
  - 随机森林
      - 不仅随机选择样本,还随机选择部分特征
      - 规避了强势样本也削弱了强势特征的影响
- 03_sc.py
  - 人工分类绘制散点图
- 04_lr.py
  - 逻辑分类 本质是线性回归
  - 用sigmoid函数将连续函数离散化
- 05_mlr.py
  - 多元分类
  - 通过对多个结果训练多个模型最终选择概率最高的类别作为预测结果
- 06_nb.py
  - 朴素贝叶斯分类器
  - 特征之间无因果关系
## day04
- 01_tts.py
  - 分类问题数据集划分
  - 测试集训练集划分API
  - 对模型进行交叉验证
  - 4种交叉验证指标
- 02_cm.py
  - 输出混淆矩阵
  - 行代表实际类别
  - 列代表预测类别
- 03_cr.py
  - 输出分类报告
- 04_car.py
  - 整理数据集
  - 交叉验证
  - 构建随机森林分类器模型
  - 使用相同标签分类器对测试数据分类
  - 得到测试分类报告
- 05_vc.py
  - 验证曲线
  - 根据API选择最优超参数
  - 绘制超参数与模型性能关系图
- 06_lc.py
  - 学习曲线
  - 根据API获取最优训练集大小
  - 绘制模型性能关系图
- 07_svm.py
  - 支持向量机
  - linea核不进行升维,无法处理先行不可分情况
  - poly和rbf将原样本进行升维增加新的特征寻求最分类边界
## day05
- 01_balance.py
  - 样本类别均衡化
  - 增加占比较小样本的权重,减少占比较大样本的权重
- 02_prob.py
  - 根据样本离分类边界的远近量化置信度,越远置信度越高
- 03_gridSearch.py
  - 网格搜索
  - 可以对多个超参数进行组合验证获得最优组合
- 04_event.py
  - 案例应用
  - 步骤：
    1. 获取数据,整理数据
    2. 整理输入集与输出集
    3. learning curve 学习曲线从而拆分训练集与输出集
    4. 选择模型
    5. 模型参数的调整,验证曲线,网络搜索,如果仍得不到较好的模型,那么需要从样本的角度进行优化
    6. 基于训练集训练, 基于测试集测试
    7. 模型评估 回归：r2_score() 分类：混淆矩阵、分类报告
    8. 如果模型不符合要求,返回第4步
    9. 模型上线,业务应用
- 05_traffic,py
  - 车流量预测
- 06_Kmeans.py
  - K均值聚类器
- 07_img.py
  - 通过K均值聚类量化图像中的颜色
