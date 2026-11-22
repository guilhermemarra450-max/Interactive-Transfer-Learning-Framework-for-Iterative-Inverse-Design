Fig. S9 · XGBoost 训练程序

一、在 Mac 上启动（建议使用带 Tkinter 的 Python 3.11 或 3.12）
解压整个压缩包，保留 S9_XGBoost.py、requirements.txt、示例 CSV 和 rf_resources 的相对位置。
打开“终端”，输入 cd 加一个空格，把解压后的 S9_XGBoost 文件夹拖入终端，按回车。
首次使用时逐行执行：

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python S9_XGBoost.py

以后只需进入此目录，执行最后的启动命令前先运行 source .venv/bin/activate。
也可以用已安装上述依赖的解释器直接运行 S9_XGBoost.py。
若 Python 提示缺少 _tkinter，可使用 python.org 的 macOS Python 3.11 或 3.12 安装包，把虚拟环境建在带 Tkinter 的解释器下。可用 python3 -m tkinter 检查窗口能否打开。

二、用论文 Table S7 的 11 个样品试运行
1. 选择 New Training (from RF)。RF Model 应显示 20260208_025521。若为空，用 RF Folder… 指向包内 rf_resources 中的 RF_for_XGBoost_20260208_025521 文件夹。
2. 在 New Data File 右边点击 Browse...，选择 S7_initial_11.csv。Sample 是样品编号，AdsSTP 是实测 CO₂ 吸附量。
3. 首次试运行可保留 Top Features = 7，保持 Pseudo-Labels 未勾选，再点击 Start Training。Top Features = 7 会使用全部 7 个输入变量；改为 4 时使用 RF 重要性排名前四的 SSA、V、Vmicro、r。此设置只控制输入列的选择与排序。
4. 完成后，在程序旁的 bestmodel 文件夹可找到本次模型与配套信息，在 data2evaluation 中可查看本次结果。各次运行按时间分别保存。
5. 若在本版本上继续训练，选择 Continue Training (from existing XGBoost)，选择 bestmodel 中生成的 .pkl 和新的带标签数据文件，并设定要追加的轮数。请将 model_info 文件与 .pkl 保存在同一目录。使用缺少输入变换记录的旧模型续训时，变换的一致性无法核实，程序会在日志中提示。

三、结果含义
输出的 R²、RMSE、MAE 是模型对本次训练样品的拟合指标。论文中的交叉验证指标需要单独计算。启用小样本评估得到的是训练集诊断。运行成功只表明恢复的程序能够完成这一流程，无法据此确认历史实验用的 Top Features 或论文中的性能数值。
原程序提供无实测标签时调用 RF 生成伪标签的分支，沿用 RF 模型本身的输出尺度。这些输出不能当作实测吸附量。随包的 11 个样品都有实测标签，直接使用 AdsSTP 训练。
