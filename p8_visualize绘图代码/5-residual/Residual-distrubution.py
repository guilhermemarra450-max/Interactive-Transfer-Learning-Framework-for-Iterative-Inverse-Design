import sys
import io
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy import stats

# 尝试导入seaborn以获取丰富的调色板（若未安装则回退）
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn未安装，将使用matplotlib默认颜色循环。")

# ======================== 全局字体设置（解决中文显示） ========================
plt = matplotlib.pyplot
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

# ======================== 语言字典 ========================
TEXTS = {
    'zh': {
        'window_title': '残差分析工具 - 残差分布图与散点图',
        'open_csv': '打开CSV文件',
        'save_fig': '保存当前图表',
        'hist_bins': '直方图分组数 (bins):',
        'show_kde': '显示核密度估计曲线 (KDE)',
        'show_normal': '显示理论正态曲线',
        'stats_title': '残差统计信息',
        'lang_switch': 'English / 中文',
        'color_scheme': '配色方案',
        'sample_size': '样本量',
        'mean': '均值',
        'std': '标准差',
        'skewness': '偏度',
        'kurtosis': '峰度',
        'min': '最小值',
        'max': '最大值',
        'shapiro_title': 'Shapiro-Wilk p值',
        'normality_conclusion_accept': '结论: 不能拒绝正态性假设 (p>0.05)',
        'normality_conclusion_reject': '结论: 拒绝正态性假设 (p<0.05)',
        'normality_failed': '正态检验失败 (样本量过大或scipy错误)',
        'hist_title': '残差分布图 (密度直方图)',
        'scatter_title': '残差 vs 预测值 散点图',
        'xlabel_residual': '残差',
        'ylabel_density': '密度',
        'xlabel_predicted': '预测值',
        'ylabel_residual': '残差',
        'hist_label': '残差直方图',
        'kde_label': 'KDE 密度曲线',
        'normal_label': '理论正态分布',
        'zero_line': 'y=0 参考线',
        'info_tip': '提示：可使用matplotlib工具栏缩放/平移',
        'load_success': '成功加载数据，共 {} 个样本',
        'load_error_cols': 'CSV文件必须包含 \'predicted_values\' 和 \'residuals\' 列',
        'load_error_empty': 'CSV文件无有效数据',
        'load_file_error': '无法读取文件：\n{}',
        'save_success': '图片已保存至：{}',
    },
    'en': {
        'window_title': 'Residual Analysis Tool - Residual Distribution & Scatter',
        'open_csv': 'Open CSV File',
        'save_fig': 'Save Figure',
        'hist_bins': 'Histogram bins:',
        'show_kde': 'Show KDE Curve',
        'show_normal': 'Show Theoretical Normal Curve',
        'stats_title': 'Residual Statistics',
        'lang_switch': 'English / 中文',
        'color_scheme': 'Color Scheme',
        'sample_size': 'Sample size',
        'mean': 'Mean',
        'std': 'Std. deviation',
        'skewness': 'Skewness',
        'kurtosis': 'Kurtosis',
        'min': 'Minimum',
        'max': 'Maximum',
        'shapiro_title': 'Shapiro-Wilk p-value',
        'normality_conclusion_accept': 'Conclusion: Cannot reject normality (p>0.05)',
        'normality_conclusion_reject': 'Conclusion: Reject normality (p<0.05)',
        'normality_failed': 'Normality test failed (large sample or scipy error)',
        'hist_title': 'Residual Distribution (Density Histogram)',
        'scatter_title': 'Residuals vs Predicted Values',
        'xlabel_residual': 'Residual',
        'ylabel_density': 'Density',
        'xlabel_predicted': 'Predicted value',
        'ylabel_residual': 'Residual',
        'hist_label': 'Residual histogram',
        'kde_label': 'KDE density',
        'normal_label': 'Theoretical normal',
        'zero_line': 'y=0 reference',
        'info_tip': 'Tip: Use matplotlib toolbar to zoom/pan',
        'load_success': 'Successfully loaded {} samples',
        'load_error_cols': 'CSV must contain \'predicted_values\' and \'residuals\' columns',
        'load_error_empty': 'No valid data in CSV',
        'load_file_error': 'Failed to read file:\n{}',
        'save_success': 'Figure saved to: {}',
    }
}

# 可用的配色方案（根据提供的txt）
COLOR_SCHEMES = [
    'deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind',
    'Set1', 'Set2', 'tab10',
    'viridis', 'plasma', 'coolwarm', 'RdBu_r', 'RdYlBu', 'seismic'
]

def get_main_color(scheme):
    """根据配色方案名称返回一个主色（用于直方图和散点图）"""
    if HAS_SEABORN:
        try:
            if scheme in ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind',
                          'Set1', 'Set2', 'tab10']:
                palette = sns.color_palette(scheme)
                return palette[0]  # 返回第一个颜色
            else:
                # 对于colormap，取中间位置的颜色
                cmap = plt.get_cmap(scheme)
                return cmap(0.5)
        except Exception:
            pass
    # 回退：使用matplotlib内置tab10或colormap
    if scheme in ['Set1', 'Set2', 'tab10']:
        cmap = plt.get_cmap('tab10')
        return cmap(0)
    elif scheme in ['viridis', 'plasma', 'coolwarm', 'RdBu_r', 'RdYlBu', 'seismic']:
        cmap = plt.get_cmap(scheme)
        return cmap(0.5)
    else:
        # 默认蓝色
        return 'steelblue'

# ======================== 默认CSV数据（与原代码相同） ========================
DEFAULT_CSV_TEXT = """predicted_values,residuals
0.24706413392078605,-0.003320243821679919
-0.5158226054825774,-0.0035503462247075124
0.03941467786191529,0.0005082915580643527
0.4877370648713918,0.21010927590391315
-1.237062780721648,-0.4948839951152988
0.6055324454593475,0.04837913101950886
0.7598244071167809,-1.3322676295501878e-15
0.442266001040003,0.21164557543885332
-0.9265018611003637,0.9265018611003637
0.5856604579694566,1.9984014443252818e-15
0.8226085805077682,0.2904416230167449
0.005890951277084459,-0.00046577085047114393
-1.2519211070621268,-0.09250760903390565
-0.011114863592496315,0.011114863592496315
-0.8979514793444499,-0.018875575835637703
-0.39231534305973376,7.46197353629352e-06
-0.9431899980173075,-0.12319102541806104
-1.404950772528493,-0.020775090166666788
1.206477246054793,-0.001336379539652155
-0.616090479612714,0.009299366462599323
0.024156570071968857,0.0005378855839361235
-0.0021201271561844238,-0.0005571989252909785
0.7311830522034221,-0.7311830522034221
0.37249853172532077,-7.216449660063518e-16
-0.47913155400445595,-0.2167599122337167
-0.3658026481456988,0.3658026481456988
-0.06033384453455271,-0.18487307187958235
0.6367758564207506,0.7105881809139598
0.6196557753528547,-0.15408953385037777
0.6011358196315306,0.3059261799111479
0.7445473488744925,0.0495053657445077
0.6933667803907901,0.1527524888314703
-0.39247038628766917,-3.316432683919679e-06
0.20495389366812528,0.7052897389114601
0.18682078136823554,-0.0009330540218948358
0.5888927420503085,-0.8633444887057451
-0.6011823142624904,0.0732974285515241
0.6566877764852039,-0.24525002003531982
0.2783036077248299,-0.5122749324538727
-0.278917788259588,0.0037688496305382513
-1.2519211070621268,-0.09250760903390565
0.3864338423605981,-0.1160504761084239
0.8346866189023406,0.14491981718706204
0.4308475617637341,0.23324874702862203
0.7338491697519882,-0.13473422126396828
0.1955661593439898,0.09267545304014363
-0.9177723788721557,-0.4444112525728574
0.8944886100320639,0.08511782605733875
-0.26564744592969985,8.326672684688674e-16
0.6762366189819066,0.14068113909403113
0.8361673239989242,-0.0062440890638617574
-1.1640126219566922,-0.3596455663082452
0.40563999500534864,0.04916713833889097
0.01325448405452374,0.00025187890699269674
-0.069892883776315,-0.0019175741465524282
0.4638202751152193,0.008294839173814461
0.298681863804914,-0.004864905309353351
1.5117884744801615,0.09716064948261116
-0.069892883776315,-0.0019175741465524282
0.17996423229087166,0.6726147035611183
0.49890879903479174,0.31800895904114596
-0.6574547816763081,-2.4424906541753444e-15
-0.018937945510988125,0.018937945510988125
0.739357328646908,0.19940723444447284
0.6516499374181131,-0.08483156225633692
-1.16971455117738,0.05346831849258371
-0.8979514793444499,-0.018875575835637703
-0.23953059668431054,9.71445146547012e-16
-0.08419507613054458,-0.010962665592527274
-1.070753333082099,-0.06714464614190363
0.45171181718347225,0.04948370670929586
-0.2039468324225745,0.2039468324225745
0.051889442798964236,0.020841515420818288
-1.2367119326211096,0.0005583087430161005
0.4638202751152193,0.008294839173814461
0.046126194473615253,0.008750848469212036
-0.06688096992689821,0.0003415855652252886
-0.03130341370775646,0.03130341370775646
-1.2047290544202964,-0.0001690950310231365
-0.24318833667882686,5.551115123125783e-16
-0.022037798957474437,0.02756745408478018
-0.9725263941190042,-0.15334471225874202
-1.404950772528493,-0.020775090166666788
-0.24318833667882686,5.551115123125783e-16
0.8312726764954261,-0.01761310109023273
-0.8617125951038055,-0.2559288925114719
-1.404950772528493,-0.020775090166666788
0.6138640737071404,-1.0096511802608377
-0.6888460389377693,-0.11070251137498521
-0.13971225795255102,0.00025054566234108955
0.5606181275277389,-0.024145763482290694
-0.6417514858379275,7.89110343651922e-05
-0.207134206662624,-0.014452567255803489
-1.2899761848154025,1.2899761848154025
-0.32799706072937346,-0.01412452490320204
-0.08591025759374593,-2.3592239273284576e-16
-0.3253001919697398,-0.14810784217878908
-0.6011823142624904,0.0732974285515241
-0.7151823908099724,-0.5508351804822342
-0.8416133484906954,0.013423532498680002
-1.070753333082099,0.0918716857512939
0.9371171903340142,0.07982676959323098
0.5081320709014023,-0.30282096070222125
1.0771989253056011,0.12804965760710596
-0.8382158300192631,0.0040566557537479175
0.03285605433655238,0.006966369099465186
-0.22687637578263284,0.3260882859149605
-1.3176045405504218,-0.1806489780383811
-0.8644499633803675,-0.6471170824798609
-0.23953059668431054,9.71445146547012e-16
-0.051983295285404116,0.051983295285404116
0.03285605433655238,0.006966369099465186
0.29327593394045726,-0.29327593394045726
-0.25862373739900446,-0.5633576194901936
-0.38492048627711056,0.00926144383875277
0.18682078136823554,-0.0009330540218948358
0.6291896800550547,-0.2943992975031694
-0.20083811739846477,-0.28552960014134177
-0.7249954755146959,-0.26078064029627746
-0.3352646934928954,-0.4698711838452871
-0.6574547816763081,-2.4424906541753444e-15
-0.2842802626909152,-1.0547118733938987e-15
-0.8218309502559258,0.05726946348595563
0.009063371984778301,-0.0009503212632434865
0.5036216210917275,0.20964850502317778
0.31694021144623385,0.40622158082798054
0.5711695167650997,0.02449340913971587
0.20092425799066096,-0.16222305890308603
0.7693854991743062,0.5979902972778329
-0.004880012246234331,0.004880012246234331
0.37249853172532077,-7.216449660063518e-16
-0.45878898123789535,-0.0017478462633971348
-1.5873521749062882,-0.6515723019208952
-0.6389256060138735,-0.5859607530380413
1.1589451180922046,3.3365129899669768
0.3181816441731828,0.29470899635061015
0.17635100101938672,-0.03230776585886358
-0.47756538524890074,-0.12922572790121395
0.03941467786191529,0.0005082915580643527
-0.207134206662624,-0.014452567255803489
-0.527582118730427,-0.00025359980419947803
-0.678050713490067,0.678050713490067
-1.2171193620862757,0.0038344188303884863
-0.45106697462817735,-0.277921901254756
-1.7629330616046104,-0.19944008304585248
0.24706413392078605,-0.003320243821679919
-0.18125807941955827,0.18125807941955827
-0.08419507613054458,-0.010962665592527274
0.6391154508210396,0.2551944509346381
-0.2696201101141762,-0.13256877461180527
0.6427390611614152,-0.13800016656445513
-0.5088037289802672,0.5088037289802672
0.5417480295816123,-2.116929333902174
0.6465120134969663,-0.34218734782551047
0.6706118411281798,0.3564660680607854
-1.4174881987410517,-0.007848146165906345
0.6425157567574573,0.17765745175912795
0.03285605433655238,0.006966369099465186
-0.592099446744762,-0.1882611047751528
0.2934162658179476,-0.004134223816779692
0.051889442798964236,0.020841515420818288
-0.08419507613054458,-0.010962665592527274
-0.22982483291657357,0.7907867942781088
-0.8416133484906954,0.013423532498680002
-1.2367119326211096,0.0005583087430161005
-0.5276593322792763,-0.008667473144623239
0.667306035014036,-8.881784197001252e-16
-1.070753333082099,0.0918716857512939
0.7628079859106423,0.13150191584503546
1.0771989253056011,0.12804965760710596
1.231108827579404,0.0007596297288319231
0.10068235430816484,0.8254314006256586
-2.008856104986352,-0.03462553422292203"""

class ResidualAnalyzer:
    def __init__(self, root):
        self.root = root
        self.lang = 'zh'   # 当前语言
        self.current_color_scheme = tk.StringVar(value='deep')
        self.bins = tk.IntVar(value=30)
        self.show_kde = tk.BooleanVar(value=True)
        self.show_normal = tk.BooleanVar(value=False)

        # 加载默认数据
        self.df = self.load_default_data()
        if self.df is None or self.df.empty:
            messagebox.showerror("Error", "无法加载默认数据，程序将退出")
            sys.exit(1)

        self.create_widgets()
        self.update_all_texts()   # 初始化界面语言
        self.update_plots()

    def load_default_data(self):
        try:
            data_io = io.StringIO(DEFAULT_CSV_TEXT)
            df = pd.read_csv(data_io)
            df = df.dropna().reset_index(drop=True)
            return df if len(df) > 0 else None
        except Exception as e:
            print("加载默认数据失败:", e)
            return None

    def load_csv_file(self):
        file_path = filedialog.askopenfilename(
            title="选择CSV文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return
        try:
            df_new = pd.read_csv(file_path)
            if 'predicted_values' not in df_new.columns or 'residuals' not in df_new.columns:
                messagebox.showerror(self.text('load_error_cols'), self.text('load_error_cols'))
                return
            df_new = df_new.dropna().reset_index(drop=True)
            if len(df_new) == 0:
                messagebox.showerror(self.text('load_error_empty'), self.text('load_error_empty'))
                return
            self.df = df_new
            self.update_plots()
            messagebox.showinfo("Success", self.text('load_success').format(len(self.df)))
        except Exception as e:
            messagebox.showerror("Error", self.text('load_file_error').format(str(e)))

    def save_figure(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG图片", "*.png"), ("JPEG图片", "*.jpg"), ("PDF文件", "*.pdf")]
        )
        if file_path:
            self.fig.savefig(file_path, dpi=150, bbox_inches='tight')
            messagebox.showinfo("Success", self.text('save_success').format(file_path))

    def text(self, key):
        """根据当前语言返回对应的字符串"""
        return TEXTS[self.lang].get(key, key)

    def toggle_language(self):
        """切换中英文并刷新界面"""
        self.lang = 'en' if self.lang == 'zh' else 'zh'
        self.update_all_texts()
        self.update_plots()   # 图表标题和轴标签也会更新

    def update_all_texts(self):
        """更新所有控件的文本（包括按钮、标签、框架标题）"""
        # 窗口标题
        self.root.title(self.text('window_title'))
        # 按钮
        self.btn_open.config(text=self.text('open_csv'))
        self.btn_save.config(text=self.text('save_fig'))
        # 标签
        self.label_bins.config(text=self.text('hist_bins'))
        self.cb_kde.config(text=self.text('show_kde'))
        self.cb_normal.config(text=self.text('show_normal'))
        self.stats_frame.config(text=self.text('stats_title'))
        self.lang_btn.config(text=self.text('lang_switch'))
        self.color_label.config(text=self.text('color_scheme'))
        self.info_label.config(text=self.text('info_tip'))
        # 刷新统计信息（里面的文字也会变成英文）
        self.update_stats()

    def create_widgets(self):
        # 主框架
        control_frame = tk.Frame(self.root, width=260, bg='#f0f0f0')
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        plot_frame = tk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- 文件操作 ---
        self.btn_open = tk.Button(control_frame, command=self.load_csv_file,
                                  bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'))
        self.btn_open.pack(pady=(10,5), padx=10, fill=tk.X)

        self.btn_save = tk.Button(control_frame, command=self.save_figure,
                                  bg='#2196F3', fg='white', font=('Arial', 10, 'bold'))
        self.btn_save.pack(pady=5, padx=10, fill=tk.X)

        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)

        # --- 直方图参数 ---
        self.label_bins = tk.Label(control_frame, bg='#f0f0f0', anchor='w')
        self.label_bins.pack(pady=(10,0), padx=10, fill=tk.X)
        bins_scale = tk.Scale(control_frame, from_=5, to=100, orient=tk.HORIZONTAL,
                              variable=self.bins, resolution=1, command=lambda x: self.update_plots())
        bins_scale.pack(padx=10, fill=tk.X)

        # KDE 和正态曲线复选框
        self.cb_kde = tk.Checkbutton(control_frame, variable=self.show_kde,
                                     command=self.update_plots, bg='#f0f0f0', anchor='w')
        self.cb_kde.pack(pady=(10,0), padx=10, fill=tk.X)

        self.cb_normal = tk.Checkbutton(control_frame, variable=self.show_normal,
                                        command=self.update_plots, bg='#f0f0f0', anchor='w')
        self.cb_normal.pack(pady=(5,0), padx=10, fill=tk.X)

        # --- 配色方案选择 ---
        self.color_label = tk.Label(control_frame, bg='#f0f0f0', anchor='w')
        self.color_label.pack(pady=(10,0), padx=10, fill=tk.X)
        color_combo = ttk.Combobox(control_frame, textvariable=self.current_color_scheme,
                                   values=COLOR_SCHEMES, state='readonly')
        color_combo.pack(pady=(2,5), padx=10, fill=tk.X)
        color_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plots())

        # --- 语言切换按钮 ---
        self.lang_btn = tk.Button(control_frame, command=self.toggle_language,
                                  bg='#FFC107', fg='black', font=('Arial', 9))
        self.lang_btn.pack(pady=(10,5), padx=10, fill=tk.X)

        # --- 统计信息框 ---
        self.stats_frame = tk.LabelFrame(control_frame, bg='#f0f0f0', padx=5, pady=5)
        self.stats_frame.pack(pady=(10,10), padx=10, fill=tk.BOTH, expand=True)
        self.stats_text = tk.Text(self.stats_frame, height=12, width=30, bg='#ffffff', font=('Consolas', 9))
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        scroll = tk.Scrollbar(self.stats_text, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 提示信息
        self.info_label = tk.Label(control_frame, bg='#f0f0f0', fg='gray', font=('Arial', 8))
        self.info_label.pack(side=tk.BOTTOM, pady=10)

        # --- 绘图区域 ---
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_stats(self):
        """更新统计信息（中英文由当前语言决定）"""
        residuals = self.df['residuals']
        n = len(residuals)
        mean_r = residuals.mean()
        std_r = residuals.std(ddof=1)
        skewness = residuals.skew()
        kurt = residuals.kurtosis()
        try:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            p_text = f"{self.text('shapiro_title')}: {shapiro_p:.4f}\n"
            if shapiro_p > 0.05:
                normality_text = self.text('normality_conclusion_accept')
            else:
                normality_text = self.text('normality_conclusion_reject')
        except Exception:
            p_text = ""
            normality_text = self.text('normality_failed')
        min_r = residuals.min()
        max_r = residuals.max()

        stats_str = f"{self.text('sample_size')}: {n}\n" \
                    f"{self.text('mean')}: {mean_r:.6f}\n" \
                    f"{self.text('std')}: {std_r:.6f}\n" \
                    f"{self.text('skewness')}: {skewness:.4f}\n" \
                    f"{self.text('kurtosis')}: {kurt:.4f}\n" \
                    f"{self.text('min')}: {min_r:.4f}\n" \
                    f"{self.text('max')}: {max_r:.4f}\n\n" \
                    f"{p_text}{normality_text}"

        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats_str)

    def update_plots(self):
        if self.df is None or self.df.empty:
            return

        residuals = self.df['residuals']
        predicted = self.df['predicted_values']
        bins = self.bins.get()
        show_kde = self.show_kde.get()
        show_normal = self.show_normal.get()
        color_scheme = self.current_color_scheme.get()
        main_color = get_main_color(color_scheme)

        self.ax1.clear()
        self.ax2.clear()

        # ---------- 子图1：残差分布 ----------
        self.ax1.hist(residuals, bins=bins, density=True, alpha=0.6,
                      color=main_color, edgecolor='black', linewidth=0.5,
                      label=self.text('hist_label'))
        if show_kde:
            try:
                kde = stats.gaussian_kde(residuals)
                x_range = np.linspace(residuals.min(), residuals.max(), 500)
                kde_vals = kde(x_range)
                self.ax1.plot(x_range, kde_vals, 'r-', linewidth=2, label=self.text('kde_label'))
            except Exception:
                pass
        if show_normal:
            mu, std = residuals.mean(), residuals.std(ddof=0)
            x_norm = np.linspace(residuals.min(), residuals.max(), 500)
            norm_pdf = stats.norm.pdf(x_norm, mu, std)
            self.ax1.plot(x_norm, norm_pdf, 'k--', linewidth=2, label=self.text('normal_label'))

        self.ax1.set_xlabel(self.text('xlabel_residual'))
        self.ax1.set_ylabel(self.text('ylabel_density'))
        self.ax1.set_title(self.text('hist_title'))
        self.ax1.legend(loc='best', fontsize=8)
        self.ax1.grid(True, linestyle=':', alpha=0.7)

        # ---------- 子图2：残差散点图 ----------
        self.ax2.scatter(predicted, residuals, s=20, alpha=0.7,
                         c=[main_color], edgecolors='k', linewidth=0.3)
        self.ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label=self.text('zero_line'))
        self.ax2.set_xlabel(self.text('xlabel_predicted'))
        self.ax2.set_ylabel(self.text('ylabel_residual'))
        self.ax2.set_title(self.text('scatter_title'))
        self.ax2.legend(loc='best', fontsize=8)
        self.ax2.grid(True, linestyle=':', alpha=0.7)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw_idle()

        # 更新统计信息（语言可能已改变）
        self.update_stats()

if __name__ == "__main__":
    root = tk.Tk()
    app = ResidualAnalyzer(root)
    root.mainloop()