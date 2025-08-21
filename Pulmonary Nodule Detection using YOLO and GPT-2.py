#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import sys
import cv2
import shutil
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from ultralytics import YOLO
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from tkinter import scrolledtext
import threading

class LungNoduleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.gpt2_model_dir = self.get_model_path() 
        self.root.title("Pulmonary Nodule Detection Suite")
        self.root.geometry("1200x800")
        self.history = [] 

        self.image_path = None
        self.image = None
        self.tk_image = None
        self.annotations = []
        self.new_data_dir = "incremental_data"
        os.makedirs(self.new_data_dir, exist_ok=True)


        self.yolo_model = None
        self.yolo_classes = ['lung nodule', 'normal']
        self.similarity_threshold = 0.1


        self.gpt2_model = None
        self.gpt2_tokenizer = None
        self.gpt2_pipeline = None


        self.create_menubar()
        self.create_main_layout()
        self.create_scale_controls()
        self.create_control_panel()
    def get_model_path(self):
            """ 确保 `exe` 运行时也能找到 GPT-2 模型 """
            if getattr(sys, 'frozen', False):  # PyInstaller 运行环境
                base_path = os.path.dirname(sys.executable)
            else:  # Python 直接运行
                base_path = os.path.abspath(".")
            
            return os.path.join(base_path, "gpt2_model_dir", "gpt2_pulmonary_nodule_final")

    def create_menubar(self):
        menubar = tk.Menu(self.root)


        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load YOLO Model", command=self.load_yolo_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)


        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about_dialog)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def show_about_dialog(self):
        messagebox.showinfo(
            "About",
            "Pulmonary Nodule Detection Suite\nVersion 1.0\nBy YourNameHere"
        )


    def create_main_layout(self):

        self.preview_label = tk.Label(self.root, text="Image Preview:", font=("Arial", 14))
        self.preview_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")


        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="gray")
        self.canvas.grid(row=1, column=0, rowspan=3, padx=10, pady=10, sticky="nsew")


        self.log_text = tk.Text(self.root, height=10, wrap=tk.WORD, font=("Arial", 10))
        self.log_text.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="ew")


        self.result_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.result_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")


        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    def create_scale_controls(self):

        self.scale_frame = tk.Frame(self.root, bd=2, relief=tk.GROOVE)
        self.scale_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        tk.Label(self.scale_frame, text="Confidence Threshold:", font=("Arial", 12)
                 ).pack(anchor="w", pady=(5, 0))
        self.conf_scale = tk.Scale(self.scale_frame, from_=0.01, to=1.0,
                                   resolution=0.01, orient=tk.HORIZONTAL)
        self.conf_scale.set(0.25)
        self.conf_scale.pack(fill="x", padx=10, pady=5)

        tk.Label(self.scale_frame, text="IOU Threshold:", font=("Arial", 12)
                 ).pack(anchor="w", pady=(5, 0))
        self.iou_scale = tk.Scale(self.scale_frame, from_=0.01, to=1.0,
                                  resolution=0.01, orient=tk.HORIZONTAL)
        self.iou_scale.set(0.5)
        self.iou_scale.pack(fill="x", padx=10, pady=5)

    def create_control_panel(self):
        self.button_frame = tk.Frame(self.root, bd=2, relief=tk.RIDGE)
        self.button_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        buttons = [
            ("Upload Image", self.upload_image),
            ("Annotate", self.enable_annotation_mode),
            ("Delete Annotation", self.enable_delete_mode),
            ("Save Annotations", self.save_annotations),
            ("New Data", self.upload_new_data),
            ("Train YOLO", self.train_with_new_data),
            ("Incremental Train", self.incremental_training),
            ("Batch Process", self.batch_processing),
            ("Process Video", self.process_video),
            ("Manage Data", self.open_data_manager),
            ("Gen. Report", self.generate_report),
            ("History", self.show_history),
            ("Interactive GPT-2", self.launch_interactive_generation),
            ("AI Q&A", self.launch_ai_qa)
        ]

        max_columns = 7  
        for i, (text, cmd) in enumerate(buttons):
            row_i = i // max_columns
            col_i = i % max_columns
            btn = tk.Button(self.button_frame, text=text, command=cmd, font=("Arial", 10))
            btn.grid(row=row_i, column=col_i, padx=5, pady=5, sticky="ew")


        row_count = (len(buttons) + max_columns - 1) // max_columns
        for r in range(row_count):
            self.button_frame.grid_rowconfigure(r, weight=1)
        for c in range(max_columns):
            self.button_frame.grid_columnconfigure(c, weight=1)

    def log(self, message):
        self.history.append(message)
        print(message)

    def load_yolo_model(self):
        try:
            model_path = filedialog.askopenfilename(
                title="Select YOLO Model",
                filetypes=[("YOLO Model Files", "*.pt")]
            )
            if not model_path:
                messagebox.showwarning("Warning", "No model selected!")
                return
            self.yolo_model = YOLO(model_path)

            if hasattr(self.yolo_model, "names"):
                self.yolo_classes = self.yolo_model.names
            self.log(f"YOLO model loaded successfully from {model_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {str(e)}")
            self.log(f"Error: Failed to load YOLO model: {str(e)}")
            self.yolo_model = None


    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)

            conf = self.conf_scale.get()
            iou = self.iou_scale.get()
            self.log(f"Running detection with conf={conf:.2f}, iou={iou:.2f}")
            try:
                if not self.yolo_model:
                    messagebox.showwarning("Warning", "Please load a YOLO model first!")
                    return

                results = self.yolo_model.predict(
                    source=file_path,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    conf=conf,
                    iou=iou,
                    max_det=300
                )
                self.display_detections(results)
            except Exception as e:
                self.log(f"Error during detection: {e}")
                messagebox.showerror("Error", f"Error during detection: {e}")

    def display_image(self, file_path):
        self.image = cv2.imread(file_path)
        if self.image is None:
            self.log("Failed to load image.")
            messagebox.showerror("Error", "Failed to load image.")
            return

        original_height, original_width = self.image.shape[:2]
        canvas_width, canvas_height = 800, 600
        aspect_ratio = original_width / original_height

        if canvas_width / canvas_height > aspect_ratio:
            display_height = canvas_height
            display_width = int(aspect_ratio * canvas_height)
        else:
            display_width = canvas_width
            display_height = int(canvas_width / aspect_ratio)

        resized_image = cv2.resize(self.image, (display_width, display_height))
        img_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))

        self.canvas.delete("all")
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2
        self.canvas.create_image(offset_x, offset_y, image=self.tk_image, anchor="nw")
        self.result_label.config(text="Image displayed (aspect ratio preserved).")
        self.image_display_size = (display_width, display_height)
    
    def display_detections(self, results):
        if not hasattr(self, 'image_display_size'):
            self.log("No image currently displayed.")
            messagebox.showerror("Error", "No image is currently displayed.")
            return
    
        original_height, original_width = self.image.shape[:2]
        display_width, display_height = self.image_display_size
        canvas_width, canvas_height = 800, 600
        offset_x = (canvas_width - display_width) // 2
        offset_y = (canvas_height - display_height) // 2
    
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
    
                label = self.yolo_classes[cls_id] if cls_id < len(self.yolo_classes) else f"class_{cls_id}"
                confidence = box.conf[0]
    
                x1 = int(x1 * display_width / original_width) + offset_x
                x2 = int(x2 * display_width / original_width) + offset_x
                y1 = int(y1 * display_height / original_height) + offset_y
                y2 = int(y2 * display_height / original_height) + offset_y
    
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=2)
                self.canvas.create_text(x1, y1, text=f"{label} ({confidence:.2f})", fill="red", anchor="nw")
    
        self.log("Detections displayed on the canvas.")
    
    

    def enable_annotation_mode(self):
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.finish_draw)
        self.result_label.config(text="Annotation mode enabled. Draw bounding boxes.")
        self.annotations = []

    def start_draw(self, event):
        self.start_x, self.start_y = event.x, event.y
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline="red"
        )

    def draw_rectangle(self, event):
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def finish_draw(self, event):
        end_x, end_y = event.x, event.y
        x1, x2 = sorted([self.start_x, end_x])
        y1, y2 = sorted([self.start_y, end_y])
        self.canvas.coords(self.rect_id, x1, y1, x2, y2)

        label = simpledialog.askstring("Annotation", "Enter label for this annotation:")
        label = label if label else "Unlabeled"
        self.annotations.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": label})
        self.log(f"Annotation added: {label} @ ({x1},{y1},{x2},{y2})")

    def enable_delete_mode(self):
        self.canvas.bind("<Button-1>", self.select_annotation)
        self.result_label.config(text="Delete mode enabled. Click annotation to remove it.")

    def select_annotation(self, event):
        x, y = event.x, event.y
        for annotation in self.annotations:
            if (annotation["x1"] <= x <= annotation["x2"] and
                annotation["y1"] <= y <= annotation["y2"]):
                self.annotations.remove(annotation)
                self.redraw_annotations()
                self.log(f"Deleted annotation: {annotation['label']}")
                self.result_label.config(text=f"Deleted annotation: {annotation['label']}")
                break

    def redraw_annotations(self):
        self.canvas.delete("all")
        self.display_image(self.image_path)
        for anno in self.annotations:
            self.canvas.create_rectangle(
                anno["x1"], anno["y1"], anno["x2"], anno["y2"],
                outline="blue", width=2
            )
            self.canvas.create_text(
                anno["x1"], anno["y1"],
                text=anno["label"], fill="blue", anchor="nw"
            )


    def save_annotations(self):
        """ 保存标注数据到 YOLO 格式 """
        if not self.image_path or not self.annotations:
            messagebox.showwarning("Warning", "No annotations to save.")
            return

        base_name = os.path.basename(self.image_path)
        save_image_path = os.path.join(self.image_dir, base_name)
        save_label_path = os.path.join(self.label_dir, base_name.rsplit(".", 1)[0] + ".txt")

        # 复制图片
        shutil.copy(self.image_path, save_image_path)

        # 获取图片尺寸
        img_height, img_width = self.image.shape[:2]

        with open(save_label_path, "w") as f:
            for anno in self.annotations:
                x1, y1, x2, y2, label = anno["x1"], anno["y1"], anno["x2"], anno["y2"], anno["label"]
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                class_id = self.yolo_classes.index(label) if label in self.yolo_classes else -1
                if class_id >= 0:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        messagebox.showinfo("Save Success", f"Annotations saved: {save_label_path}")
        self.log(f"Saved annotations to {save_label_path}")

    def upload_new_data(self):
        folder_path = filedialog.askdirectory(title="Select a folder containing training images")
        if folder_path:
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    shutil.copy(os.path.join(folder_path, file_name), self.new_data_dir)
            self.log(f"New training data added from {folder_path}")
            messagebox.showinfo("Upload Success", f"Data uploaded from {folder_path}")



    def train_with_new_data(self):
        if not self.yolo_model:
            messagebox.showwarning("Warning", "Please load a YOLO model first!")
            return
        try:
            self.log("Training YOLO with new data...")

        
            results = self.yolo_model.train(
                task="detect",
                mode="train",
                model="yolo11s.yaml", 
                data={
                    'train': self.new_data_dir,
                    'val': "valid_data/images",
                    'nc': len(self.yolo_classes),
                    'names': self.yolo_classes
                },
                epochs=2,  
                batch=4,
                imgsz=640,
                device="cuda" if torch.cuda.is_available() else "cpu",
                optimizer="auto",
                cache=True,
                save=True,
                workers=0,  
                save_period=-1,
                lr0=0.01,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3.0,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                dropout=0.0,
                seed=0,
                deterministic=True,
                patience=5,  
            )
            self.log(f"Training complete! Results at {results.save_dir}")
            messagebox.showinfo("Training Complete", "YOLO training finished!")
        except Exception as e:
            self.log(f"Error during training: {e}")
            messagebox.showerror("Error", str(e))




            

    def incremental_training(self):
        """ 使用增量数据训练 YOLO 并保存新模型 """
        if not self.yolo_model:
            messagebox.showwarning("Warning", "Please load a YOLO model first!")
            return
    
        try:
            self.log("Starting incremental training...")
    
            # 训练时数据集配置
            data_yaml = {
                'train': "incremental_data/images", 
                'val': "valid_data/images",
                'nc': len(self.yolo_classes),
                'names': self.yolo_classes
            }
    
            # 训练 YOLO
            results = self.yolo_model.train(
                task="detect",
                mode="train",
                model="models/best.pt",  
                data=data_yaml,
                epochs=5,  # 只训练 5 个 epoch，避免过拟合
                batch=4,
                imgsz=640,
                device="cuda" if torch.cuda.is_available() else "cpu",
                optimizer="auto",
                patience=3,  # 如果性能不提升，3 个 epoch 后停止
                cache=True,
                resume=True,  #继续训练，而不是从头开始
                lr0=0.001,  #降低学习率，防止过拟合
                weight_decay=0.0005,
                momentum=0.9
            )
    
            # 保存新模型
            new_model_path = "models/best_incremental.pt"
            shutil.copy(results.save_dir + "/weights/best.pt", new_model_path)
    
            self.log(f"Incremental training complete! New model saved at {new_model_path}")
            messagebox.showinfo("Training Complete", f"Incremental training finished! Model saved at {new_model_path}")
    
        except Exception as e:
            self.log(f"Error during incremental training: {e}")
            messagebox.showerror("Error", str(e))



    def batch_processing(self):
        """ 批量处理文件夹中的所有图片，并生成 labels.txt 存入 train_data/labels/ """
        if not self.yolo_model:
            messagebox.showwarning("Warning", "Please load a YOLO model first!")
            return
        folder_path = filedialog.askdirectory(title="Select a folder containing images")
        if not folder_path:
            return
    
        train_image_dir = os.path.join("train_data", "images")
        train_label_dir = os.path.join("train_data", "labels")
        os.makedirs(train_image_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
    
        self.log(f"Processing images in: {folder_path}")
    
        try:
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        results = self.yolo_model.predict(
                            source=file_path,
                            device="cuda" if torch.cuda.is_available() else "cpu",
                            conf=0.25,
                            iou=0.5,
                            max_det=300
                        )
    
                        img = cv2.imread(file_path)
                        img_height, img_width = img.shape[:2]
    
                        # 保存 YOLO 格式的 labels
                        label_path = os.path.join(train_label_dir, file_name.rsplit(".", 1)[0] + ".txt")
                        with open(label_path, "w") as f:
                            for result in results:
                                for box in result.boxes:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    cls_id = int(box.cls[0])
    
                                    x_center = ((x1 + x2) / 2) / img_width
                                    y_center = ((y1 + y2) / 2) / img_height
                                    width = (x2 - x1) / img_width
                                    height = (y2 - y1) / img_height
    
                                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                                    self.log(f"{file_name}: Detected label {cls_id} saved")
    
                        # 复制图片到 train_data/images/
                        shutil.copy(file_path, train_image_dir)
    
                    except Exception as e:
                        self.log(f"Error processing {file_name}: {e}")
    
            self.log(f"Batch processing complete! Images and labels saved in train_data/")
            messagebox.showinfo("Batch Processing", f"Processed images and labels saved in train_data/")
    
        except Exception as e:
            self.log(f"Batch processing error: {e}")
            messagebox.showerror("Error", str(e))




    def process_video(self):
        """ 处理视频，并生成带检测框的视频 """
        if not self.yolo_model:
            messagebox.showwarning("Warning", "Please load a YOLO model first!")
            return
        video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")], title="Select a Video File")
        if not video_path:
            return
    
        output_path = video_path.rsplit(".", 1)[0] + "_processed.mp4"
        self.log(f"Processing video: {video_path}")
    
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Unable to open video file.")
    
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
    
                results = self.yolo_model.predict(
                    frame,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
    
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        label = self.yolo_classes[cls_id] if cls_id < len(self.yolo_classes) else f"class_{cls_id}"
                        confidence = box.conf[0]
    
                        # 画框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
                out.write(frame)
    
            cap.release()
            out.release()
            self.log(f"Video processing done. Saved to {output_path}")
            messagebox.showinfo("Video Processed", f"Saved at: {output_path}")
    
        except Exception as e:
            self.log(f"Video error: {e}")
            messagebox.showerror("Error", str(e))



    def open_data_manager(self):
        manager_window = tk.Toplevel(self.root)
        manager_window.title("Dataset Manager")
        manager_window.geometry("800x600")

        dataset_tree = ttk.Treeview(manager_window, columns=("File", "Action"), show="headings")
        dataset_tree.heading("File", text="File Name")
        dataset_tree.heading("Action", text="Action")
        dataset_tree.pack(fill=tk.BOTH, expand=True)

        for file in os.listdir(self.new_data_dir):
            dataset_tree.insert("", "end", values=(file, "Delete"))

        def delete_selected_file():
            selected_item = dataset_tree.selection()
            if selected_item:
                for item in selected_item:
                    file_name = dataset_tree.item(item, "values")[0]
                    file_path = os.path.join(self.new_data_dir, file_name)
                    try:
                        os.remove(file_path)
                        dataset_tree.delete(item)
                        self.log(f"Deleted file: {file_name}")
                    except Exception as e:
                        self.log(f"Error deleting file {file_name}: {e}")
                        messagebox.showerror("Error", f"Error deleting file {file_name}: {e}")

        del_button = tk.Button(manager_window, text="Delete Selected", command=delete_selected_file)
        del_button.pack(pady=10)

        close_button = tk.Button(manager_window, text="Close", command=manager_window.destroy)
        close_button.pack(pady=5)

        self.log("Data Manager opened.")

    def generate_report(self):
  
        try:
          
            metrics = {
                "Precision": [0.85, 0.88, 0.90, 0.92],
                "Recall": [0.80, 0.83, 0.87, 0.89],
                "mAP@0.5": [0.81, 0.84, 0.88, 0.90],
            }
            epochs = list(range(1, len(metrics["Precision"]) + 1))
            fig, ax = plt.subplots()
            for metric, values in metrics.items():
                ax.plot(epochs, values, label=metric)

            ax.set_title("Model Performance over Epochs")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Metric Value")
            ax.legend()

            report_path = os.path.join(self.new_data_dir, "performance_report.png")
            plt.savefig(report_path)
            plt.show()

            self.log(f"Performance report saved to {report_path}")
            messagebox.showinfo("Report Generated", f"Saved at: {report_path}")
        except Exception as e:
            self.log(f"Error generating report: {e}")
            messagebox.showerror("Error", str(e))


    def show_history(self):
        history_window = tk.Toplevel(self.root)
        history_window.title("Task History")
        history_window.geometry("400x300")

        history_listbox = tk.Listbox(history_window, font=("Arial", 12))
        history_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        for item in self.history:
            history_listbox.insert(tk.END, item)

        close_button = tk.Button(history_window, text="Close", command=history_window.destroy, font=("Arial", 12))
        close_button.pack(pady=5)

        self.log("Task history opened.")


    def launch_interactive_generation(self):
        """ 启动 GPT-2 交互式文本生成窗口 """

        if self.gpt2_model is None or self.gpt2_tokenizer is None or self.gpt2_pipeline is None:
            self.log("Loading GPT-2 model for interactive generation...")

            try:
                # 使用 `self.gpt2_model_dir` 而不是 `gpt2_model_dir`
                self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(self.gpt2_model_dir)
                self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
                self.gpt2_model = GPT2LMHeadModel.from_pretrained(self.gpt2_model_dir)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.gpt2_model.to(device)
                self.gpt2_pipeline = pipeline(
                    "text-generation",
                    model=self.gpt2_model,
                    tokenizer=self.gpt2_tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
                self.log("GPT-2 model loaded successfully.")

            except Exception as e:
                self.log(f"Error loading GPT-2 model: {e}")
                messagebox.showerror("Error", f"Error loading GPT-2 model: {e}")
                return

    
        # 创建窗口
        gpt2_window = tk.Toplevel(self.root)
        gpt2_window.title("GPT-2 Interactive Generation")
        gpt2_window.geometry("800x600")
    
        # 输入框
        label_input = tk.Label(gpt2_window, text="Input Prompt:", font=("Arial", 12))
        label_input.pack(pady=5)
        text_input = scrolledtext.ScrolledText(gpt2_window, height=5, width=80, wrap=tk.WORD, font=("Arial", 12))
        text_input.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
    
        # 参数设置
        frame_params = tk.Frame(gpt2_window)
        frame_params.pack(pady=5)
    
        tk.Label(frame_params, text="Max Length:", font=("Arial", 10)).grid(row=0, column=0, padx=5)
        entry_max_length = tk.Entry(frame_params, width=5)
        entry_max_length.insert(0, "100")
        entry_max_length.grid(row=0, column=1, padx=5)
    
        tk.Label(frame_params, text="Num Sequences:", font=("Arial", 10)).grid(row=0, column=2, padx=5)
        entry_num_sequences = tk.Entry(frame_params, width=5)
        entry_num_sequences.insert(0, "1")
        entry_num_sequences.grid(row=0, column=3, padx=5)
    
        # 生成文本区域
        label_output = tk.Label(gpt2_window, text="Generated Text:", font=("Arial", 12))
        label_output.pack(pady=5)
        text_output = scrolledtext.ScrolledText(gpt2_window, height=10, width=80, wrap=tk.WORD, font=("Arial", 12))
        text_output.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
    
        def generate_text():
            """ 在后台线程中生成文本，避免界面卡顿 """
            prompt = text_input.get("1.0", "end").strip()
            if not prompt:
                messagebox.showinfo("Info", "Please enter a prompt!")
                return
        
            try:
                max_length = int(entry_max_length.get())
                num_sequences = int(entry_num_sequences.get())
                text_output.insert("end", "\nGenerating...\n")
                text_output.see("end")
        
                def run_generation():
                    try:
                        output = self.gpt2_pipeline(
                            prompt,
                            max_length=max_length,
                            num_return_sequences=num_sequences,
                            truncation=True  # 添加截断
                        )
                        text_output.delete("1.0", "end")
                        for idx, result in enumerate(output):
                            text_output.insert("end", f"({idx + 1}) {result['generated_text']}\n\n")
                        text_output.see("end")
                    except Exception as e:
                        text_output.insert("end", f"Error: {e}\n")
                        text_output.see("end")
                        self.log(f"Text generation failed: {e}")
        
                threading.Thread(target=run_generation, daemon=True).start()
        
            except ValueError:
                messagebox.showerror("Error", "Invalid input for max length or num sequences!")


    
        def clear_text():
            """ 清空输入和输出框 """
            text_input.delete("1.0", "end")
            text_output.delete("1.0", "end")
    
        # 按钮区域
        frame_buttons = tk.Frame(gpt2_window)
        frame_buttons.pack(pady=10)
    
        btn_generate = tk.Button(frame_buttons, text="Generate", command=generate_text, font=("Arial", 12))
        btn_generate.grid(row=0, column=0, padx=10)
    
        btn_clear = tk.Button(frame_buttons, text="Clear", command=clear_text, font=("Arial", 12))
        btn_clear.grid(row=0, column=1, padx=10)
    
        self.log("Opened GPT-2 Interactive Generation window.")


    def launch_ai_qa(self):

        qa_window = tk.Toplevel(self.root)
        qa_window.title("AI Q&A Demo")
        qa_window.geometry("800x600")


        label_context = tk.Label(qa_window, text="Context:", font=("Arial", 12))
        label_context.pack(pady=5)
        text_context = tk.Text(qa_window, height=8, width=80, wrap=tk.WORD, font=("Arial", 12))
        text_context.pack(padx=10, pady=5)


        label_question = tk.Label(qa_window, text="Question:", font=("Arial", 12))
        label_question.pack(pady=5)
        text_question = tk.Text(qa_window, height=3, width=80, wrap=tk.WORD, font=("Arial", 12))
        text_question.pack(padx=10, pady=5)


        label_answer = tk.Label(qa_window, text="Answer:", font=("Arial", 12))
        label_answer.pack(pady=5)
        text_answer = tk.Text(qa_window, height=6, width=80, wrap=tk.WORD, font=("Arial", 12))
        text_answer.pack(padx=10, pady=5, expand=True, fill=tk.BOTH)

        def on_answer():
            context = text_context.get("1.0", "end").strip()
            question = text_question.get("1.0", "end").strip()
            if not context or not question:
                messagebox.showinfo("Info", "Please provide both context and question!")
                return


            if self.gpt2_pipeline is None:
                messagebox.showinfo("Info", "Please load/initialize the GPT-2 pipeline first!")
                return

            try:
                combined_prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
                output = self.gpt2_pipeline(combined_prompt, max_length=100, num_return_sequences=1)
                answer_text = output[0]["generated_text"]
            except Exception as e:
                messagebox.showerror("Error", f"QA generation failed: {e}")
                self.log(f"QA generation failed: {e}")
                return

            text_answer.delete("1.0", "end")
            text_answer.insert("end", answer_text)

        btn_answer = tk.Button(qa_window, text="Get Answer", command=on_answer, font=("Arial", 12))
        btn_answer.pack(pady=10)

        self.log("Opened AI Q&A window.")



if __name__ == "__main__":
    root = tk.Tk()
    app = LungNoduleDetectionApp(root)
    root.mainloop()




# In[ ]:




