import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
import threading

class InpaintGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("图像修复系统")
        self.root.geometry("1000x800")

        # 初始化变量
        self.image_path = ""
        self.mask_path = ""
        self.output_paths = {
            "masked": "image_masked.jpg",
            "output": "output.jpg"
        }

        # 创建界面布局
        self.create_widgets()

        # 配置网格布局权重
        for i in range(4):
            self.root.grid_rowconfigure(i, weight=1)
            self.root.grid_columnconfigure(i, weight=1)

    def create_widgets(self):
        """创建界面组件"""
        # 文件选择区域
        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.grid(row=0, column=0, columnspan=4, pady=10)

        # 图像显示区域
        self.create_image_displays()

        # 按钮区域
        self.create_buttons()

    def create_image_displays(self):
        """创建图像显示区域"""
        # 左上：原始图像
        self.img_label = tk.Label(self.root, text="原始图像预览区", relief="groove")
        self.img_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # 右上：掩码图像
        self.mask_label = tk.Label(self.root, text="掩码图像预览区", relief="groove")
        self.mask_label.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # 左下：带掩码图像
        self.masked_label = tk.Label(self.root, text="掩码覆盖预览区", relief="groove")
        self.masked_label.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        # 右下：修复结果
        self.output_label = tk.Label(self.root, text="修复结果预览区", relief="groove")
        self.output_label.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

    def create_buttons(self):
        """创建按钮区域"""
        # 文件选择按钮
        self.btn_select_image = ttk.Button(
            self.btn_frame,
            text="选择图像文件",
            command=lambda: self.select_file("image")
        )
        self.btn_select_image.pack(side=tk.LEFT, padx=10)

        self.btn_select_mask = ttk.Button(
            self.btn_frame,
            text="选择掩码文件",
            command=lambda: self.select_file("mask")
        )
        self.btn_select_mask.pack(side=tk.LEFT, padx=10)

        # 生成按钮
        self.btn_process = ttk.Button(
            self.root,
            text="开始修复",
            command=self.start_inpaint_thread,
            style="Accent.TButton"
        )
        self.btn_process.grid(row=3, column=0, columnspan=2, pady=20)

        # 进度条
        self.progress = ttk.Progressbar(
            self.root,
            orient=tk.HORIZONTAL,
            mode='indeterminate'
        )
        self.progress.grid(row=4, column=0, columnspan=2, sticky="ew", padx=20)

    def select_file(self, file_type):
        """文件选择对话框"""
        file_path = filedialog.askopenfilename(
            title=f"选择{'图像' if file_type == 'image' else '掩码'}文件",
            filetypes=[("JPEG文件", "*.jpg"), ("所有文件", "*.*")]
        )

        if file_path:
            if file_type == "image":
                self.image_path = file_path
                self.update_preview(file_path, self.img_label)
            else:
                self.mask_path = file_path
                self.update_preview(file_path, self.mask_label)

    def update_preview(self, path, label):
        """更新预览图像"""
        try:
            img = Image.open(path)
            img.thumbnail((400, 400))  # 限制预览尺寸
            photo = ImageTk.PhotoImage(img)
            label.configure(image=photo)
            label.image = photo  # 保持引用
            label.configure(text="")
        except Exception as e:
            self.show_error(f"无法加载图像: {str(e)}")

    def start_inpaint_thread(self):
        """启动修复线程"""
        if not self.validate_inputs():
            return

        # 禁用按钮
        self.btn_process["state"] = "disabled"
        self.progress.start()

        # 创建线程
        thread = threading.Thread(target=self.run_inpaint)
        thread.start()
        self.root.after(100, self.check_thread, thread)

    def check_thread(self, thread):
        """检查线程状态"""
        if thread.is_alive():
            self.root.after(100, self.check_thread, thread)
        else:
            self.progress.stop()
            self.btn_process["state"] = "normal"
            self.update_results()

    def run_inpaint(self):
        """执行修复算法"""
        from inpaint import inpaint  # 导入修复函数

        try:
            # 调用修复函数并传递用户选择的路径
            inpaint(self.image_path, self.mask_path)
        except Exception as e:
            self.show_error(f"修复失败: {str(e)}")

    def update_results(self):
        """更新结果预览"""
        for key, label in zip(["masked", "output"], [self.masked_label, self.output_label]):
            path = self.output_paths[key]
            if os.path.exists(path):
                self.update_preview(path, label)

    def validate_inputs(self):
        """验证输入有效性"""
        errors = []
        if not self.image_path:
            errors.append("请选择图像文件")
        if not self.mask_path:
            errors.append("请选择掩码文件")

        if errors:
            self.show_error("\n".join(errors))
            return False
        return True

    def show_error(self, message):
        """显示错误弹窗"""
        tk.messagebox.showerror("错误", message)


if __name__ == "__main__":
    root = tk.Tk()

    app = InpaintGUI(root)
    root.mainloop()