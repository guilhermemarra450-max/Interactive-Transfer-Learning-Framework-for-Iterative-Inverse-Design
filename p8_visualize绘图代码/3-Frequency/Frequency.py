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

class StatFreqGUI:
    def __init__(self, root, data=None):
        self.root = root
        self.root.title("统计类型可视化工具 - 特征频次分析")
        self.root.geometry("1360x750")

        if data is not None:
            self.df = data
        else:
            self.df = self.load_default_data()

        # 所有可供选择的特征列（全部列）
        self.all_columns = list(self.df.columns)

        # 主框架
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板
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

        # 图形和画布
        self.fig = plt.figure(figsize=(9, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 底部按钮区域
        btn_frame = ttk.Frame(plot_frame)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="保存图片", command=self.save_figure_with_options).pack(side=tk.LEFT, padx=5)

        # -------------------- 交互变量 --------------------
        self.plot_type = tk.StringVar(value="频次条形图")
        self.palette = tk.StringVar(value="deep")
        self.english_mode = tk.BooleanVar(value=False)
        self.selected_column = tk.StringVar(value=self.all_columns[0] if self.all_columns else "")
        self.sort_by = tk.StringVar(value="按值排序")
        self.font_size = tk.IntVar(value=10)

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
                                       values=['频次条形图', '频次折线图', '散点图', '饼图'],
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

        # 字体大小调节
        font_frame = ttk.Frame(scrollable_frame)
        font_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        ttk.Label(font_frame, text="图表字体大小:").pack(side=tk.LEFT, padx=5)
        self.font_spin = ttk.Spinbox(font_frame, from_=8, to=20, increment=1, textvariable=self.font_size, width=5)
        self.font_spin.pack(side=tk.LEFT, padx=5)
        self.font_spin.bind('<KeyRelease>', lambda e: self.update_plot())
        self.font_spin.bind('<Button-4>', lambda e: self.update_plot())
        self.font_spin.bind('<Button-5>', lambda e: self.update_plot())
        row += 1

        # 特征列选择
        col_frame = ttk.LabelFrame(scrollable_frame, text="选择要统计的特征列 (横坐标)")
        col_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        ttk.Label(col_frame, text="特征列:").pack(side=tk.LEFT, padx=5)
        self.col_combo = ttk.Combobox(col_frame, textvariable=self.selected_column,
                                      values=self.all_columns, state='readonly', width=30)
        self.col_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.col_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plot())
        row += 1

        # 排序方式选择
        sort_frame = ttk.Frame(scrollable_frame)
        sort_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        ttk.Label(sort_frame, text="排序方式:").pack(side=tk.LEFT, padx=5)
        self.sort_combo = ttk.Combobox(sort_frame, textvariable=self.sort_by,
                                       values=['按值排序', '按频次降序'], state='readonly', width=15)
        self.sort_combo.pack(side=tk.LEFT, padx=5)
        self.sort_combo.bind('<<ComboboxSelected>>', lambda e: self.update_plot())
        row += 1

        # 说明文字
        self.help_label = ttk.Label(scrollable_frame, text="",
                                    font=('Arial', 8), foreground='gray', justify=tk.LEFT, wraplength=420)
        self.help_label.grid(row=row, column=0, columnspan=2, pady=10, padx=5)
        row += 1

        # 重新加载文件按钮
        ttk.Button(scrollable_frame, text="重新加载CSV文件", command=self.reload_data).grid(row=row, column=0, columnspan=2, pady=10)

        # 初始化帮助文本和绘图
        self.update_help_text()
        self.update_plot()

    # ---------- 默认数据 ----------
    def load_default_data(self):
        csv_str = """Feature,Score,Age,Category,Region
A,85,25,High,North
B,90,30,High,South
C,78,22,Medium,East
D,92,35,High,West
E,88,28,Medium,North
F,76,26,Low,South
G,89,29,High,East
H,84,27,Medium,West
I,91,31,High,North
J,79,24,Low,South"""
        df = pd.read_csv(StringIO(csv_str), skipinitialspace=True)
        return df

    def toggle_language(self):
        self.english_mode.set(not self.english_mode.get())
        if self.english_mode.get():
            self.lang_btn.config(text="English")
            self.plot_combo['values'] = ['Frequency Bar', 'Frequency Line', 'Scatter Plot', 'Pie Chart']
            current = self.plot_type.get()
            mapping = {'频次条形图':'Frequency Bar', '频次折线图':'Frequency Line', '散点图':'Scatter Plot', '饼图':'Pie Chart'}
            if current in mapping:
                self.plot_type.set(mapping[current])
        else:
            self.lang_btn.config(text="中文")
            self.plot_combo['values'] = ['频次条形图', '频次折线图', '散点图', '饼图']
            current = self.plot_type.get()
            reverse_mapping = {'Frequency Bar':'频次条形图', 'Frequency Line':'频次折线图', 'Scatter Plot':'散点图', 'Pie Chart':'饼图'}
            if current in reverse_mapping:
                self.plot_type.set(reverse_mapping[current])
        self.update_help_text()
        self.update_plot()

    def update_help_text(self):
        if self.english_mode.get():
            help_txt = """Instructions:
- Select a feature column as X-axis (the column to count frequencies).
- The Y-axis shows the frequency (count) of each distinct value.
- Supports bar, line, scatter, and pie charts.
- Sorting: by value order or descending frequency.
- X-axis ticks are formatted to 1 decimal place for numeric columns.
- Use the font size control to adjust chart text sizes.
- Color scheme applies globally."""
        else:
            help_txt = """说明：
- 选择一个特征列作为横坐标（需统计频次的列）。
- 纵坐标显示该列每个不同取值出现的频次（计数）。
- 支持条形图、折线图、散点图、饼图。
- 排序方式：按原始值排序 或 按频次降序。
- 横坐标刻度：若为数值列，保留一位小数。
- 使用字体大小控件调节图表文字大小。
- 配色方案全局应用。"""
        self.help_label.config(text=help_txt)

    def get_freq_data(self):
        col = self.selected_column.get()
        if not col or col not in self.df.columns:
            return [], [], False
        series = self.df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)

        freq = series.value_counts()
        sort_mode = self.sort_by.get()
        if sort_mode == "按值排序":
            if is_numeric:
                freq = freq.sort_index(key=lambda x: x.astype(float))
            else:
                freq = freq.sort_index()
        else:
            freq = freq.sort_values(ascending=False)

        values = freq.index.tolist()
        counts = freq.values.tolist()
        return values, counts, is_numeric

    def format_tick_label(self, value, is_numeric):
        if is_numeric:
            try:
                return f"{float(value):.1f}"
            except (ValueError, TypeError):
                return str(value)
        else:
            return str(value)

    def get_color_list(self, n_colors):
        palette_name = self.palette.get()
        try:
            colors = sns.color_palette(palette_name, n_colors)
        except ValueError:
            cmap = plt.get_cmap(palette_name)
            colors = [cmap(i / max(1, n_colors-1)) for i in range(n_colors)]
        return colors

    def update_plot(self):
        values, counts, is_numeric = self.get_freq_data()
        english = self.english_mode.get()
        plot_type = self.plot_type.get()
        if english:
            type_mapping = {'频次条形图':'Frequency Bar', '频次折线图':'Frequency Line', '散点图':'Scatter Plot', '饼图':'Pie Chart'}
            plot_type = type_mapping.get(plot_type, plot_type)
        else:
            type_mapping = {'Frequency Bar':'频次条形图', 'Frequency Line':'频次折线图', 'Scatter Plot':'散点图', 'Pie Chart':'饼图'}
            plot_type = type_mapping.get(plot_type, plot_type)

        self.ax.clear()
        self.ax.set_aspect('auto')

        if not values:
            msg = "没有可用的数据，请检查所选列是否为空" if not english else "No valid data, check if selected column is empty"
            self.ax.text(0.5, 0.5, msg, ha='center', va='center', transform=self.ax.transAxes, fontsize=self.font_size.get())
            self.ax.set_axis_off()
            self.canvas.draw()
            return

        n = len(values)
        colors = self.get_color_list(n)
        x_pos = np.arange(n)
        fontsize_base = self.font_size.get()
        tick_labels = [self.format_tick_label(v, is_numeric) for v in values]

        # ---------- 绘图分支 ----------
        if plot_type in ["频次条形图", "Frequency Bar"]:
            bars = self.ax.bar(x_pos, counts, color=colors)
            self.ax.set_xticks(x_pos)
            self.ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=fontsize_base-1)
            self.ax.tick_params(axis='y', labelsize=fontsize_base-1)
            for bar, cnt in zip(bars, counts):
                self.ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                             str(cnt), ha='center', va='bottom', fontsize=fontsize_base-2)

        elif plot_type in ["频次折线图", "Frequency Line"]:
            self.ax.plot(x_pos, counts, marker='o', color=colors[0] if colors else 'blue', linewidth=2)
            self.ax.fill_between(x_pos, 0, counts, alpha=0.2, color=colors[0] if colors else 'blue')
            self.ax.set_xticks(x_pos)
            self.ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=fontsize_base-1)
            self.ax.tick_params(axis='y', labelsize=fontsize_base-1)
            for i, cnt in enumerate(counts):
                self.ax.text(i, cnt + 0.5, str(cnt), ha='center', va='bottom', fontsize=fontsize_base-2)

        elif plot_type in ["散点图", "Scatter Plot"]:
            self.ax.scatter(x_pos, counts, color=colors[0] if colors else 'blue', s=80, zorder=5)
            self.ax.plot(x_pos, counts, linestyle='--', color='gray', alpha=0.5, zorder=2)
            self.ax.set_xticks(x_pos)
            self.ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=fontsize_base-1)
            self.ax.tick_params(axis='y', labelsize=fontsize_base-1)
            for i, cnt in enumerate(counts):
                self.ax.text(i, cnt + 0.5, str(cnt), ha='center', va='bottom', fontsize=fontsize_base-2)

        elif plot_type in ["饼图", "Pie Chart"]:
            if n > 10:
                top10_idx = np.argsort(counts)[-10:]
                top10_values = [values[i] for i in top10_idx]
                top10_counts = [counts[i] for i in top10_idx]
                other_count = sum(counts) - sum(top10_counts)
                if other_count > 0:
                    top10_values.append("其他")
                    top10_counts.append(other_count)
                values, counts = top10_values, top10_counts
                n = len(values)
                tick_labels = [self.format_tick_label(v, is_numeric) for v in values]
                colors = self.get_color_list(n)

            wedges, texts, autotexts = self.ax.pie(counts, labels=tick_labels, autopct='%1.1f%%',
                                                   colors=colors, textprops={'fontsize': fontsize_base-1})
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(fontsize_base-2)
            self.ax.axis('equal')
            col_name = self.selected_column.get()
            # 简化饼图标题
            if english:
                title = f"Frequency Distribution of '{col_name}'"
            else:
                title = f"'{col_name}' 的频次分布"
            self.ax.set_title(title, fontsize=fontsize_base+2)
            self.fig.tight_layout()
            self.canvas.draw()
            return

        else:
            self.ax.text(0.5, 0.5, "不支持的绘图类型", ha='center', va='center', fontsize=fontsize_base)
            self.canvas.draw()
            return

        # 设置标题和轴标签（非饼图）- 简化标题，去除绘图类型和配色信息
        col_name = self.selected_column.get()
        if english:
            xlabel = col_name
            ylabel = "Frequency"
            title = f"Frequency Distribution of '{col_name}'"
        else:
            xlabel = col_name
            ylabel = "频次"
            title = f"'{col_name}' 的频次分布"

        self.ax.set_xlabel(xlabel, fontsize=fontsize_base)
        self.ax.set_ylabel(ylabel, fontsize=fontsize_base)
        self.ax.set_title(title, fontsize=fontsize_base+2)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.fig.tight_layout()
        self.canvas.draw()

    # ---------- 保存图片 ----------
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
            if new_df.empty or new_df.shape[1] < 1:
                messagebox.showerror("错误", "CSV文件至少需要一列数据")
                return
            self.df = new_df
            self.all_columns = list(self.df.columns)
            self.col_combo['values'] = self.all_columns
            if self.all_columns:
                self.selected_column.set(self.all_columns[0])
            else:
                self.selected_column.set("")
            self.update_help_text()
            self.update_plot()
            msg = "已加载：" if not self.english_mode.get() else "Loaded: "
            messagebox.showinfo("成功" if not self.english_mode.get() else "Success",
                                f"{msg}{os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"加载失败: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = StatFreqGUI(root)
    root.mainloop()