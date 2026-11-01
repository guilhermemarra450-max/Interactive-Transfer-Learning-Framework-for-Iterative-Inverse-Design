import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DistributionGUI:
    def __init__(self, root, data):
        self.root = root
        self.root.title("数据分布可视化工具 (支持中英文)")
        self.root.geometry("1300x750")
        self.df = data
        # 仅保留数值列，避免绘图错误
        self.variables = list(self.df.select_dtypes(include=[np.number]).columns)
        if not self.variables:
            messagebox.showerror("错误", "数据中无数值列，无法绘图！")
            self.root.destroy()
            return

        # 主框架
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板（滚动）
        control_canvas = tk.Canvas(main_frame, width=320, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=control_canvas.yview)
        scrollable_frame = ttk.Frame(control_canvas)
        scrollable_frame.bind("<Configure>", lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all")))
        control_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=scrollbar.set)
        control_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        # 右侧绘图区域
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 固定图形和轴（不使用 tight_layout）
        self.fig = Figure(figsize=(9, 6), dpi=100)
        self.ax_pos = [0.08, 0.08, 0.88, 0.88]
        self.ax = self.fig.add_axes(self.ax_pos)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 底部按钮
        btn_frame = ttk.Frame(plot_frame)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="保存图片", command=self.save_figure).pack(side=tk.LEFT, padx=5)

        # ========== 控制变量 ==========
        self.plot_type = tk.StringVar(value="直方图+KDE")
        self.palette = tk.StringVar(value="deep")
        self.reverse_palette = tk.BooleanVar(value=False)   # 新增：反转颜色顺序
        self.bins = tk.IntVar(value=30)
        self.english_mode = tk.BooleanVar(value=False)

        # ========== 构建控件 ==========
        row = 0

        # 中英文切换
        lang_frame = ttk.Frame(scrollable_frame)
        lang_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        ttk.Label(lang_frame, text="界面语言 (绘图文本):").pack(side=tk.LEFT, padx=5)
        self.lang_btn = ttk.Button(lang_frame, text="中文", width=6, command=self.toggle_language)
        self.lang_btn.pack(side=tk.LEFT, padx=5)
        row += 1

        # 绘图类型选择
        ttk.Label(scrollable_frame, text="绘图类型:").grid(row=row, column=0, sticky=tk.W, pady=5, padx=5)
        plot_combo = ttk.Combobox(scrollable_frame, textvariable=self.plot_type,
                                  values=['直方图+KDE', '箱线图', '小提琴图'],
                                  state='readonly')
        plot_combo.grid(row=row, column=1, pady=5, padx=5, sticky=tk.W)
        plot_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plot())
        row += 1

        # 配色方案选择
        ttk.Label(scrollable_frame, text="配色方案:").grid(row=row, column=0, sticky=tk.W, pady=5, padx=5)
        palette_combo = ttk.Combobox(scrollable_frame, textvariable=self.palette,
                                     values=[
                                         'deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind',
                                         'viridis', 'plasma', 'coolwarm', 'RdBu_r', 'RdYlBu', 'seismic',
                                         'Set1', 'Set2', 'tab10'
                                     ],
                                     state='readonly')
        palette_combo.grid(row=row, column=1, pady=5, padx=5, sticky=tk.W)
        palette_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plot())
        row += 1

        # 反转颜色选项（颜色顺序可选）
        reverse_frame = ttk.Frame(scrollable_frame)
        reverse_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=2, padx=5)
        ttk.Checkbutton(reverse_frame, text="反转颜色顺序", variable=self.reverse_palette,
                        command=self.update_plot).pack(side=tk.LEFT)
        row += 1

        # 变量选择
        var_frame = ttk.LabelFrame(scrollable_frame, text="变量选择")
        var_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        self.var_all_btn = ttk.Button(var_frame, text="全选", command=self.select_all_vars)
        self.var_all_btn.pack(side=tk.LEFT, padx=5, pady=2)
        self.var_none_btn = ttk.Button(var_frame, text="取消全选", command=self.select_none_vars)
        self.var_none_btn.pack(side=tk.LEFT, padx=5, pady=2)

        # 变量复选框容器（保存引用，便于重新加载时重建）
        self.var_check_frame = ttk.Frame(var_frame)
        self.var_check_frame.pack(fill=tk.X, padx=5, pady=5)
        self.var_vars = {}
        self._build_var_checkboxes()   # 根据当前 self.variables 构建
        row += 1

        # bins 调节
        bins_frame = ttk.Frame(scrollable_frame)
        bins_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        ttk.Label(bins_frame, text="直方图 bins 数:").pack(side=tk.LEFT)
        bins_spin = ttk.Spinbox(bins_frame, from_=5, to=100, increment=1, textvariable=self.bins, width=6,
                                command=self.update_plot)
        bins_spin.pack(side=tk.LEFT, padx=5)
        bins_spin.bind('<KeyRelease>', lambda e: self.update_plot())
        row += 1

        # 重新加载文件
        ttk.Button(scrollable_frame, text="重新加载CSV文件", command=self.reload_data).grid(row=row, column=0, columnspan=2, pady=10, padx=5)
        row += 1

        # 说明文字
        help_text = """说明：
- 直方图+KDE：显示每个变量的分布及核密度估计
- 箱线图：展示中位数、四分位数及异常值
- 小提琴图：结合箱线图和核密度形状
- 支持多选变量，并可自由切换配色方案
- 勾选「反转颜色顺序」可改变变量间颜色分配顺序
- 自动过滤非数值列，点击「重新加载CSV文件」可更换数据"""
        ttk.Label(scrollable_frame, text=help_text,
                  font=('Arial', 8), foreground='gray', justify=tk.LEFT).grid(row=row, column=0, columnspan=2, pady=10, padx=5)

        self.update_plot()

    def _build_var_checkboxes(self):
        """根据 self.variables 重建变量复选框（仅数值列）"""
        # 清空原有控件
        for widget in self.var_check_frame.winfo_children():
            widget.destroy()
        self.var_vars.clear()
        for i, var in enumerate(self.variables):
            var_var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(self.var_check_frame, text=var, variable=var_var, command=self.update_plot)
            cb.grid(row=i//2, column=i%2, sticky=tk.W, padx=5)
            self.var_vars[var] = var_var

    def toggle_language(self):
        self.english_mode.set(not self.english_mode.get())
        self.lang_btn.config(text="English" if self.english_mode.get() else "中文")
        self.update_plot()

    def select_all_vars(self):
        for var in self.var_vars:
            self.var_vars[var].set(True)
        self.update_plot()

    def select_none_vars(self):
        for var in self.var_vars:
            self.var_vars[var].set(False)
        self.update_plot()

    def get_selected_vars(self):
        return [var for var, v in self.var_vars.items() if v.get()]

    def get_color_list(self, n_colors):
        palette_name = self.palette.get()
        try:
            colors = sns.color_palette(palette_name, n_colors)
        except ValueError:
            # 使用 matplotlib colormap
            cmap = plt.get_cmap(palette_name)
            if n_colors == 1:
                colors = [cmap(0.5)]
            else:
                colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
        # 根据反转标志决定颜色顺序
        if self.reverse_palette.get():
            colors = list(reversed(colors))
        return colors

    def update_plot(self):
        selected = self.get_selected_vars()
        if not selected:
            self.ax.clear()
            msg = "请至少选择一个变量" if not self.english_mode.get() else "Please select at least one variable"
            self.ax.text(0.5, 0.5, msg, ha='center', va='center', transform=self.ax.transAxes, fontsize=14)
            self.ax.set_axis_off()
            self.canvas.draw()
            return

        plot_type = self.plot_type.get()
        self.ax.clear()
        colors = self.get_color_list(len(selected))
        english = self.english_mode.get()

        if english:
            plot_type_en = {
                '直方图+KDE': 'Histogram + KDE',
                '箱线图': 'Boxplot',
                '小提琴图': 'Violin Plot'
            }.get(plot_type, plot_type)
            title = f"Data Distribution - {plot_type_en}  (Colormap: {self.palette.get()})"
            if self.reverse_palette.get():
                title += " [Reversed]"
            xlabel = "Value"
            ylabel_hist = "Frequency"
            ylabel_box_violin = "Value"
        else:
            title = f"数据分布图 - {plot_type}  (配色: {self.palette.get()})"
            if self.reverse_palette.get():
                title += " [反转]"
            xlabel = "值"
            ylabel_hist = "频数"
            ylabel_box_violin = "数值"

        if plot_type == '直方图+KDE':
            for idx, var in enumerate(selected):
                sns.histplot(data=self.df, x=var, kde=True, bins=self.bins.get(),
                             alpha=0.5, color=colors[idx], label=var, ax=self.ax)
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel_hist)
            self.ax.legend()

        elif plot_type == '箱线图':
            data_to_plot = [self.df[var].dropna() for var in selected]
            bp = self.ax.boxplot(data_to_plot, tick_labels=selected,
                                 patch_artist=True, showmeans=True, meanline=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            for median in bp['medians']:
                median.set_color('black')
            for mean in bp['means']:
                mean.set_color('red')
            self.ax.set_ylabel(ylabel_box_violin)
            self.ax.tick_params(axis='x', rotation=45)

        elif plot_type == '小提琴图':
            data_to_plot = [self.df[var].dropna() for var in selected]
            parts = self.ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
            if 'cmeans' in parts:
                parts['cmeans'].set_color('red')
            if 'cmedians' in parts:
                parts['cmedians'].set_color('black')
            self.ax.set_xticks(range(1, len(selected)+1))
            self.ax.set_xticklabels(selected, rotation=45)
            self.ax.set_ylabel(ylabel_box_violin)

        self.ax.set_title(title)
        self.canvas.draw()

    def save_figure(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG图片", "*.png"), ("所有文件", "*.*")])
        if file_path:
            self.fig.savefig(file_path, dpi=150, bbox_inches='tight')
            msg = "图片已保存至：" if not self.english_mode.get() else "Image saved to: "
            messagebox.showinfo("保存成功" if not self.english_mode.get() else "Save Success",
                                f"{msg}{file_path}")

    def reload_data(self):
        file_path = filedialog.askopenfilename(title="选择CSV文件", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            new_df = pd.read_csv(file_path)
            # 仅保留数值列
            numeric_cols = new_df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                messagebox.showerror("错误", "所选文件不含数值列，无法绘图！")
                return
            # 更新数据和变量列表
            self.df = new_df
            self.variables = numeric_cols
            self._build_var_checkboxes()      # 重建复选框
            self.update_plot()                # 刷新图形
            msg = "已加载：" if not self.english_mode.get() else "Loaded: "
            messagebox.showinfo("成功" if not self.english_mode.get() else "Success",
                                f"{msg}{os.path.basename(file_path)} (数值列: {len(numeric_cols)})")
        except Exception as e:
            error_msg = "加载失败：" if not self.english_mode.get() else "Load failed: "
            messagebox.showerror("错误" if not self.english_mode.get() else "Error",
                                 f"{error_msg}{str(e)}")


if __name__ == '__main__':
    import sys

    def load_initial_data():
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
            else:
                print(f"文件不存在: {file_path}")
                root_temp = tk.Tk()
                root_temp.withdraw()
                file_path = filedialog.askopenfilename(title="选择数据文件", filetypes=[("CSV files", "*.csv")])
                root_temp.destroy()
                if not file_path:
                    print("未选择文件，退出")
                    sys.exit(1)
                df = pd.read_csv(file_path)
        else:
            if os.path.exists('data_after_imputation.csv'):
                df = pd.read_csv('data_after_imputation.csv')
            else:
                root_temp = tk.Tk()
                root_temp.withdraw()
                file_path = filedialog.askopenfilename(title="选择数据文件", filetypes=[("CSV files", "*.csv")])
                root_temp.destroy()
                if not file_path:
                    print("未选择文件，退出")
                    sys.exit(1)
                df = pd.read_csv(file_path)
        return df

    df = load_initial_data()
    # 预处理：如果初始数据无数值列则报错退出
    if df.select_dtypes(include=[np.number]).empty:
        root_temp = tk.Tk()
        root_temp.withdraw()
        messagebox.showerror("错误", "初始数据不含数值列，程序退出。")
        root_temp.destroy()
        sys.exit(1)

    root = tk.Tk()
    app = DistributionGUI(root, df)
    root.mainloop()