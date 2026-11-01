import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from io import StringIO
import os

# ---------------------------- 设置中文字体 ----------------------------
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class StatCompareGUI:
    def __init__(self, root, data=None):
        self.root = root
        self.root.title("统计比较可视化工具 - 首列作为横坐标")
        self.root.geometry("1360x750")

        if data is not None:
            self.df = data
        else:
            self.df = self.load_default_data()

        # 自动识别第一列为横坐标列（无论列名是什么）
        self.x_col_name = self.df.columns[0]   # 横坐标列名
        self.value_cols = [c for c in self.df.columns if c != self.x_col_name]

        # 主框架
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板（宽度适当增加）
        control_canvas = tk.Canvas(main_frame, width=450, highlightthickness=0)
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

        # 固定图形和轴
        self.fig = plt.figure(figsize=(9, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 底部按钮区域
        btn_frame = ttk.Frame(plot_frame)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="保存图片", command=self.save_figure_with_options).pack(side=tk.LEFT, padx=5)

        # -------------------- 交互变量 --------------------
        self.plot_type = tk.StringVar(value="分组柱状图")
        self.palette = tk.StringVar(value="deep")
        self.english_mode = tk.BooleanVar(value=False)
        self.col_vars = {}

        # -------------------- 构建控件 --------------------
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
        self.plot_combo = ttk.Combobox(scrollable_frame, textvariable=self.plot_type,
                                       values=['分组柱状图', '折线图', '散点图', '单系列柱状图'],
                                       state='readonly', width=20)
        self.plot_combo.grid(row=row, column=1, pady=5, padx=5, sticky=tk.W)
        self.plot_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plot())
        row += 1

        # 配色方案选择
        ttk.Label(scrollable_frame, text="配色方案:").grid(row=row, column=0, sticky=tk.W, pady=5, padx=5)
        palette_combo = ttk.Combobox(scrollable_frame, textvariable=self.palette,
                                     values=[
                                         'deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind',
                                         'viridis', 'plasma', 'coolwarm', 'RdBu_r', 'RdYlBu', 'seismic',
                                         'Set1', 'Set2', 'tab10'
                                     ],
                                     state='readonly', width=20)
        palette_combo.grid(row=row, column=1, pady=5, padx=5, sticky=tk.W)
        palette_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plot())
        row += 1

        # 纵坐标列选择（双列显示，支持换行）
        col_frame = ttk.LabelFrame(scrollable_frame, text="纵坐标列 (可多选)")
        col_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        btn_frame_col = ttk.Frame(col_frame)
        btn_frame_col.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(btn_frame_col, text="全选", command=self.select_all_cols).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame_col, text="清空", command=self.select_none_cols).pack(side=tk.LEFT, padx=2)

        self.col_check_frame = tk.Frame(col_frame)
        self.col_check_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.refresh_col_checkboxes()   # 双列实现
        row += 1

        # 说明文字（动态更新）
        self.help_label = ttk.Label(scrollable_frame, text="",
                                    font=('Arial', 8), foreground='gray', justify=tk.LEFT, wraplength=420)
        self.help_label.grid(row=row, column=0, columnspan=2, pady=10, padx=5)
        row += 1

        # 重新加载文件按钮
        ttk.Button(scrollable_frame, text="重新加载CSV文件", command=self.reload_data).grid(row=row, column=0, columnspan=2, pady=10)

        # 初始化帮助文本和绘图
        self.update_help_text()
        self.update_plot()

    # ---------- 数据加载 ----------
    def load_default_data(self):
        # 默认数据（首列为Feature）
        csv_str = """Feature,Original_Skew,Imputed_Skew,Logged_Skew,Scaled_Skew,Original_Kurt,Imputed_Kurt,Logged_Kurt,Scaled_Kurt,Original_Mean,Imputed_Mean,Logged_Mean,Scaled_Mean,Original_Median,Imputed_Median,Logged_Median,Scaled_Median,Skewness_Change_Imputed,Skewness_Change_Logged,Skewness_Change_Scaled,Kurtosis_Change_Imputed,Kurtosis_Change_Logged,Kurtosis_Change_Scaled
T,-0.099 ,-0.093 ,-0.093 ,-0.093 ,-1.133 ,-1.102 ,-1.102 ,-1.102 ,562.557 ,561.937 ,561.937 ,0.067 ,526.850 ,526.850 ,526.850 ,0.000 ,0.007 ,0.007 ,0.007 ,0.031 ,0.031 ,0.031 
v,1.997 ,2.015 ,-0.870 ,-0.870 ,9.596 ,11.241 ,1.015 ,1.015 ,8.773 ,8.962 ,2.204 ,-0.444 ,10.000 ,10.000 ,2.403 ,0.000 ,0.018 ,2.867 ,2.867 ,1.645 ,8.581 ,8.581 
t,1.973 ,2.171 ,-0.332 ,-0.332 ,5.393 ,6.438 ,4.104 ,4.104 ,81.259 ,78.823 ,4.295 ,0.439 ,60.000 ,60.000 ,4.121 ,0.000 ,0.198 ,2.305 ,2.305 ,1.045 ,1.289 ,1.289 
r,1.375 ,1.701 ,0.737 ,0.737 ,1.870 ,3.236 ,-0.391 ,-0.391 ,0.935 ,0.850 ,0.495 ,0.129 ,0.500 ,0.500 ,0.405 ,0.000 ,0.326 ,0.638 ,0.638 ,1.366 ,2.262 ,2.262 
V,1.717 ,1.836 ,1.055 ,1.055 ,4.525 ,5.180 ,0.903 ,0.903 ,0.409 ,0.400 ,0.309 ,0.198 ,0.285 ,0.285 ,0.251 ,0.000 ,0.119 ,0.662 ,0.662 ,0.655 ,3.623 ,3.623 
Vmicro,1.606 ,1.814 ,1.184 ,1.184 ,3.557 ,4.757 ,1.876 ,1.876 ,0.282 ,0.274 ,0.230 ,0.152 ,0.230 ,0.230 ,0.207 ,0.000 ,0.208 ,0.423 ,0.423 ,1.200 ,1.681 ,1.681 
SSA,0.989 ,1.001 ,-1.809 ,-1.809 ,0.773 ,0.828 ,3.026 ,3.026 ,841.972 ,840.683 ,6.148 ,-0.373 ,740.699 ,740.699 ,6.609 ,0.000 ,0.012 ,2.799 ,2.799 ,0.054 ,2.253 ,2.253 
AdsSTP,1.698 ,1.798 ,-0.021 ,-0.021 ,10.379 ,11.808 ,1.585 ,1.585 ,3.810 ,3.801 ,1.879 ,-0.077 ,3.716 ,3.716 ,1.928 ,0.000 ,0.100 ,1.719 ,1.719 ,1.429 ,8.793 ,8.793"""
        df = pd.read_csv(StringIO(csv_str), skipinitialspace=True)
        return df

    def refresh_col_checkboxes(self):
        """依据当前数据框的第一列作为横坐标，双列显示其余列复选框"""
        for widget in self.col_check_frame.winfo_children():
            widget.destroy()
        self.col_vars.clear()
        # 更新横坐标列名和纵坐标列
        self.x_col_name = self.df.columns[0]
        self.value_cols = [c for c in self.df.columns if c != self.x_col_name]

        # 设置两列等宽
        self.col_check_frame.grid_columnconfigure(0, weight=1)
        self.col_check_frame.grid_columnconfigure(1, weight=1)

        for i, col in enumerate(self.value_cols):
            var = tk.BooleanVar(value=(i < 4))  # 默认选中前4列
            cb = tk.Checkbutton(self.col_check_frame, text=col, variable=var,
                                command=self.update_plot, wraplength=200, anchor='w', justify='left')
            row = i // 2
            column = i % 2
            cb.grid(row=row, column=column, sticky='w', padx=5, pady=2)
            self.col_vars[col] = var

    def select_all_cols(self):
        for var in self.col_vars.values():
            var.set(True)
        self.update_plot()

    def select_none_cols(self):
        for var in self.col_vars.values():
            var.set(False)
        self.update_plot()

    def get_selected_cols(self):
        return [col for col, var in self.col_vars.items() if var.get()]

    def get_color_list(self, n_colors):
        palette_name = self.palette.get()
        try:
            colors = sns.color_palette(palette_name, n_colors)
        except ValueError:
            cmap = plt.get_cmap(palette_name)
            colors = [cmap(i / max(1, n_colors-1)) for i in range(n_colors)]
        return colors

    def toggle_language(self):
        self.english_mode.set(not self.english_mode.get())
        if self.english_mode.get():
            self.lang_btn.config(text="English")
            self.plot_combo['values'] = ['Grouped Bar Chart', 'Line Chart', 'Scatter Plot', 'Single Bar Chart']
            current = self.plot_type.get()
            mapping = {'分组柱状图':'Grouped Bar Chart', '折线图':'Line Chart', '散点图':'Scatter Plot', '单系列柱状图':'Single Bar Chart'}
            if current in mapping:
                self.plot_type.set(mapping[current])
        else:
            self.lang_btn.config(text="中文")
            self.plot_combo['values'] = ['分组柱状图', '折线图', '散点图', '单系列柱状图']
            current = self.plot_type.get()
            reverse_mapping = {'Grouped Bar Chart':'分组柱状图', 'Line Chart':'折线图', 'Scatter Plot':'散点图', 'Single Bar Chart':'单系列柱状图'}
            if current in reverse_mapping:
                self.plot_type.set(reverse_mapping[current])
        self.update_help_text()
        self.update_plot()

    def update_help_text(self):
        # 动态使用实际横坐标列名
        x_name = self.x_col_name
        if self.english_mode.get():
            help_txt = f"""Instructions:
- The first column '{x_name}' is used as X-axis automatically.
- Multiple value columns can be selected for comparison.
- "Grouped Bar Chart": compare multiple columns side by side.
- "Single Bar Chart": only the first selected column.
- Line and scatter plots support multiple columns.
- Color scheme applies globally."""
        else:
            help_txt = f"""说明：
- 首列 '{x_name}' 自动作为横坐标。
- 纵坐标列可多选，若多选则绘制分组图形。
- 「分组柱状图」：多列对比；「单系列柱状图」：仅绘制第一个选中列。
- 折线图和散点图同样支持多列。
- 配色方案全局应用。"""
        self.help_label.config(text=help_txt)

    def update_plot(self):
        selected_cols = self.get_selected_cols()
        english = self.english_mode.get()

        if not selected_cols:
            self.ax.clear()
            msg = "请至少选择一个纵坐标列" if not english else "Please select at least one value column"
            self.ax.text(0.5, 0.5, msg, ha='center', va='center', transform=self.ax.transAxes, fontsize=14)
            self.ax.set_axis_off()
            self.canvas.draw()
            return

        plot_type = self.plot_type.get()
        if english:
            type_mapping = {'分组柱状图':'Grouped Bar Chart', '折线图':'Line Chart', '散点图':'Scatter Plot', '单系列柱状图':'Single Bar Chart'}
            plot_type = type_mapping.get(plot_type, plot_type)
        else:
            type_mapping = {'Grouped Bar Chart':'分组柱状图', 'Line Chart':'折线图', 'Scatter Plot':'散点图', 'Single Bar Chart':'单系列柱状图'}
            plot_type = type_mapping.get(plot_type, plot_type)

        self.ax.clear()
        # 使用第一列数据作为横坐标
        x_values = self.df[self.x_col_name].astype(str).tolist()
        x_pos = np.arange(len(x_values))
        n_series = len(selected_cols)
        colors = self.get_color_list(n_series)

        if plot_type in ["分组柱状图", "Grouped Bar Chart"]:
            width = 0.8 / n_series if n_series > 0 else 0.8
            for i, col in enumerate(selected_cols):
                y_vals = self.df[col].values
                offset = (i - n_series/2 + 0.5) * width
                self.ax.bar(x_pos + offset, y_vals, width, label=col, color=colors[i])
            self.ax.set_xticks(x_pos)
            self.ax.set_xticklabels(x_values, rotation=45, ha='right')
            self.ax.legend()

        elif plot_type in ["单系列柱状图", "Single Bar Chart"]:
            col = selected_cols[0]
            y_vals = self.df[col].values
            self.ax.bar(x_pos, y_vals, color=colors[0], label=col)
            self.ax.set_xticks(x_pos)
            self.ax.set_xticklabels(x_values, rotation=45, ha='right')
            self.ax.legend()

        elif plot_type in ["折线图", "Line Chart"]:
            for i, col in enumerate(selected_cols):
                y_vals = self.df[col].values
                self.ax.plot(x_pos, y_vals, marker='o', label=col, color=colors[i], linewidth=2)
            self.ax.set_xticks(x_pos)
            self.ax.set_xticklabels(x_values, rotation=45, ha='right')
            self.ax.legend()

        elif plot_type in ["散点图", "Scatter Plot"]:
            for i, col in enumerate(selected_cols):
                y_vals = self.df[col].values
                self.ax.scatter(x_pos, y_vals, label=col, color=colors[i], s=50)
            self.ax.set_xticks(x_pos)
            self.ax.set_xticklabels(x_values, rotation=45, ha='right')
            self.ax.legend()

        else:
            self.ax.text(0.5, 0.5, "不支持的绘图类型", ha='center', va='center')
            self.canvas.draw()
            return

        # 横坐标标签使用第一列的名称
        xlabel = self.x_col_name if english else self.x_col_name  # 中英文下均保持原列名，也可根据需要翻译，但通常列名是英文/数字，不需要翻译
        if english:
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel("Value")
            title = f"Statistical Comparison - {plot_type} (Colormap: {self.palette.get()})"
        else:
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel("数值")
            title = f"统计对比图 - {plot_type}  (配色: {self.palette.get()})"

        self.ax.set_title(title)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.fig.tight_layout()
        self.canvas.draw()

    # ---------- 保存图片（支持比例、分辨率，默认300dpi）----------
    def save_figure_with_options(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("保存图片选项")
        dialog.geometry("300x280")
        dialog.transient(self.root)
        dialog.grab_set()

        dpi_var = tk.IntVar(value=300)
        width_var = tk.DoubleVar(value=9.0)
        height_var = tk.DoubleVar(value=6.0)

        def set_ratio(ratio):
            if ratio == "16:9":
                width_var.set(16)
                height_var.set(9)
            elif ratio == "4:3":
                width_var.set(12)
                height_var.set(9)
            elif ratio == "1:1":
                width_var.set(8)
                height_var.set(8)

        row = 0
        ttk.Label(dialog, text="分辨率 (DPI):").grid(row=row, column=0, pady=5, padx=5, sticky='e')
        dpi_spin = ttk.Spinbox(dialog, from_=50, to=600, increment=10, textvariable=dpi_var, width=10)
        dpi_spin.grid(row=row, column=1, pady=5, padx=5, sticky='w')
        row += 1

        ttk.Label(dialog, text="图片宽度 (英寸):").grid(row=row, column=0, pady=5, padx=5, sticky='e')
        width_spin = ttk.Spinbox(dialog, from_=4, to=20, increment=0.5, textvariable=width_var, width=10)
        width_spin.grid(row=row, column=1, pady=5, padx=5, sticky='w')
        row += 1

        ttk.Label(dialog, text="图片高度 (英寸):").grid(row=row, column=0, pady=5, padx=5, sticky='e')
        height_spin = ttk.Spinbox(dialog, from_=3, to=15, increment=0.5, textvariable=height_var, width=10)
        height_spin.grid(row=row, column=1, pady=5, padx=5, sticky='w')
        row += 1

        ratio_frame = ttk.Frame(dialog)
        ratio_frame.grid(row=row, column=0, columnspan=2, pady=10)
        ttk.Label(ratio_frame, text="常用比例:").pack(side=tk.LEFT)
        ttk.Button(ratio_frame, text="16:9", command=lambda: set_ratio("16:9")).pack(side=tk.LEFT, padx=2)
        ttk.Button(ratio_frame, text="4:3", command=lambda: set_ratio("4:3")).pack(side=tk.LEFT, padx=2)
        ttk.Button(ratio_frame, text="1:1", command=lambda: set_ratio("1:1")).pack(side=tk.LEFT, padx=2)
        row += 1

        def do_save():
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG图片", "*.png"), ("所有文件", "*.*")])
            if not file_path:
                dialog.destroy()
                return
            try:
                current_size = self.fig.get_size_inches()
                self.fig.set_size_inches(width_var.get(), height_var.get())
                self.fig.savefig(file_path, dpi=dpi_var.get(), bbox_inches='tight')
                self.fig.set_size_inches(current_size)
                msg = "图片已保存至：" if not self.english_mode.get() else "Image saved to: "
                messagebox.showinfo("保存成功" if not self.english_mode.get() else "Save Success",
                                    f"{msg}{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {e}")
            finally:
                dialog.destroy()

        ttk.Button(dialog, text="保存", command=do_save).grid(row=row, column=0, columnspan=2, pady=10)

    def reload_data(self):
        file_path = filedialog.askopenfilename(title="选择CSV文件", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            new_df = pd.read_csv(file_path, skipinitialspace=True)
            if new_df.shape[1] < 2:
                messagebox.showerror("错误", "CSV文件至少需要两列（一列横坐标，至少一列数值）")
                return
            self.df = new_df
            # 重新识别第一列
            self.x_col_name = self.df.columns[0]
            self.value_cols = [c for c in self.df.columns if c != self.x_col_name]
            self.refresh_col_checkboxes()   # 这会重建复选框并更新数据
            self.update_help_text()          # 更新帮助文本中的横坐标名称
            self.update_plot()
            msg = "已加载：" if not self.english_mode.get() else "Loaded: "
            messagebox.showinfo("成功" if not self.english_mode.get() else "Success",
                                f"{msg}{os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"加载失败: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = StatCompareGUI(root)
    root.mainloop()