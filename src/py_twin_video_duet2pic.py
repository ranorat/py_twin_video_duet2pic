import os
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk

class MarkerData:
    def __init__(self):
        self.frame_l = 0
        self.frame_r = 0
        self.is_saved = False

class VideoPlayerPane(tk.LabelFrame):
    def __init__(self, parent, title_prefix, app_ref):
        super().__init__(parent, text=title_prefix, padx=5, pady=5)
        self.app_ref = app_ref
        self.title_prefix = title_prefix
        
        self.cap = None
        self.last_frame = None

        self.file_path = None
        self.file_basename = ""
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame_idx = 0
        self.current_angle = 0
        self.is_dragging = False
        
        self.is_playing = False
        self.play_job = None
        
        # 修正: 描画領域のサイズが画像によって勝手に拡張されないよう、サイズを固定する親フレームを配置
        canvas_container = tk.Frame(self, bg="black")
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        canvas_container.pack_propagate(False)  # 子要素(Label)のサイズ変更による親の拡大を完全に阻止する

        # 描画領域
        self.canvas_label = tk.Label(canvas_container, bg="black", anchor="center")
        self.canvas_label.pack(fill=tk.BOTH, expand=True)
        
        # コントロールUI
        row1 = tk.Frame(self)
        row1.pack(fill=tk.X, pady=2)
        self.btn_load = tk.Button(row1, text=f"{title_prefix} 読込", font=("Meiryo", 9), command=self.load_file)
        self.btn_load.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self.btn_play = tk.Button(row1, text="再生/停止", font=("Meiryo", 9), command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        
        row2 = tk.Frame(self)
        row2.pack(fill=tk.X, pady=2)
        self.btn_prev = tk.Button(row2, text="[<] コマ戻し", font=("Meiryo", 9), command=self.step_prev)
        self.btn_prev.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self.btn_next = tk.Button(row2, text="コマ送り [>]", font=("Meiryo", 9), command=self.step_next)
        self.btn_next.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        
        # スライダー
        self.slider = tk.Scale(self, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_move)
        self.slider.pack(fill=tk.X, pady=2)
        self.slider.bind("<ButtonPress-1>", self.on_slider_press)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)
        
        # 時間・フレーム表記
        self.lbl_time = tk.Label(self, text="00:00 (Frame: 0)", font=("Meiryo", 9))
        self.lbl_time.pack(pady=2)

    def load_file(self, filepath=None, is_sub_call=False):
        self.stop_play()
        if not filepath:
            filepath = filedialog.askopenfilename(
                title=f"{self.title_prefix}の動画を選択",
                filetypes=[("動画ファイル", "*.mp4 *.ts *.mpg *.mpeg"), ("すべて", "*.*")]
            )
        if not filepath:
            return
            
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            if not is_sub_call:
                messagebox.showerror("エラー", "動画ファイルを開けませんでした。")
            return
            
        self.file_path = filepath
        self.file_basename = os.path.splitext(os.path.basename(filepath))[0]
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0
            
        self.slider.config(to=max(0, self.total_frames - 1))
        self.current_frame_idx = 0
        self.seek_frame(0)
        self.app_ref.update_diff_display()

        if not is_sub_call:
            other_pane = self.app_ref.pane_r if self.title_prefix == "左" else self.app_ref.pane_l
            if other_pane.cap is None:
                if messagebox.askyesno("確認", f"もう一方の（{other_pane.title_prefix}側）に同じ動画を読み込みますか？"):
                    other_pane.load_file(filepath, is_sub_call=True)

    def seek_frame(self, frame_idx):
        if not self.cap:
            return
        self.current_frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
        
        if not self.is_dragging:
            self.slider.set(self.current_frame_idx)
        self.update_time_label()

    def display_frame(self, frame=None):
        if frame is not None:
            self.last_frame = frame
        elif self.last_frame is None:
            return

        frame_to_show = self.last_frame.copy()

        # 回転処理
        if self.current_angle == 90:
            frame_to_show = cv2.rotate(frame_to_show, cv2.ROTATE_90_CLOCKWISE)
        elif self.current_angle == 180:
            frame_to_show = cv2.rotate(frame_to_show, cv2.ROTATE_180)
        elif self.current_angle == 270:
            frame_to_show = cv2.rotate(frame_to_show, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        # RGB変換
        frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # 描画領域の現在の正確なピクセルサイズを取得してアスペクト比維持でサムネイル化
        self.update_idletasks()
        w = self.canvas_label.winfo_width()
        h = self.canvas_label.winfo_height()
        
        if w < 10: w = 300 
        if h < 10: h = 200

        # 元画像のサイズ
        img_w, img_h = img.size
        
        # 領域にぴったり収まる最大の倍率を計算（アスペクト比完全維持）
        ratio = min(w / img_w, h / img_h)
        new_w = max(1, int(img_w * ratio))
        new_h = max(1, int(img_h * ratio))
        
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(image=img_resized)
        self.canvas_label.config(image=self.photo)

    def update_time_label(self):
        secs = self.current_frame_idx / self.fps if self.fps > 0 else 0
        m = int(secs // 60)
        s = int(secs % 60)
        self.lbl_time.config(text=f"{m:02d}:{s:02d} (Frame: {self.current_frame_idx})")
        self.app_ref.update_diff_display()

    def step_next(self):
        if self.current_frame_idx < self.total_frames - 1:
            self.seek_frame(self.current_frame_idx + 1)

    def step_prev(self):
        self.stop_play()
        if self.current_frame_idx > 0:
            self.seek_frame(self.current_frame_idx - 1)

    def toggle_play(self):
        if self.is_playing:
            self.stop_play()
        else:
            if self.total_frames > 0 and self.current_frame_idx >= self.total_frames - 1:
                return
            self.app_ref.stop_sync_play()
            self.start_play()

    def start_play(self):
        if self.is_playing:
            return
        self.is_playing = True
        self.play_loop()

    def stop_play(self):
        self.is_playing = False
        if self.play_job is not None:
            self.after_cancel(self.play_job)
            self.play_job = None

    def play_loop(self):
        if not self.is_playing:
            return
            
        if self.current_frame_idx < self.total_frames - 1:
            self.step_next()
            interval = int(1000.0 / self.fps) if self.fps > 0 else 33
            self.play_job = self.after(interval, self.play_loop)
        else:
            self.stop_play()

    def on_slider_press(self, event):
        self.is_dragging = True

    def on_slider_release(self, event):
        self.is_dragging = False
        self.seek_frame(int(self.slider.get()))

    def on_slider_move(self, val):
        if self.is_dragging:
            self.seek_frame(int(float(val)))

    def rotate(self, clockwise):
        self.current_angle = (self.current_angle + (90 if clockwise else -90)) % 360
        self.seek_frame(self.current_frame_idx)


class MainApp(tk.Tk):
    # バージョン定数をクラス変数として定義
    VERSION = "1.0.0"

    def __init__(self):
        super().__init__()
        self.title("ツインビデオデュエット2Pic (Python版)")
        self.geometry("1300x810")
        
        # ウィンドウの最小サイズ（幅・高さ）を制限
        self.minsize(1150, 760)

        # ウィンドウサイズ変更監視
        self.bind("<Configure>", self.on_window_resize)

        self.markers = [MarkerData() for _ in range(5)]
        self.is_playing = False
        self.play_job = None
        
        # 最後に保存したファイル名（ベース名）を記憶する変数
        self.last_saved_filename = ""

        # 上部メニュー
        menubar = tk.Menu(self)
        help_menu = tk.Menu(menubar, tearoff=0)
        
        # バージョン情報にサードパーティのクレジットを明記
        about_text = (
            f"ツインビデオデュエット2Pic v{MainApp.VERSION}\n\n"
            "【サードパーティ・クレジット】\n"
            "・OpenCV (opencv-python) - Apache License 2.0\n"
            "・Pillow - MIT-CMU License\n\n"
            "Copyright c 2026 ranorat"
        )
        help_menu.add_command(label="バージョン情報", command=lambda: messagebox.showinfo("バージョン情報", about_text))
        menubar.add_cascade(label="ヘルプ", menu=help_menu)
        self.config(menu=menubar)

        # 左右ビューア配置用フレーム（均等サイズで伸縮）
        viewer_frame = tk.Frame(self)
        viewer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        viewer_frame.columnconfigure(0, weight=1, uniform="pane")
        viewer_frame.columnconfigure(1, weight=1, uniform="pane")
        viewer_frame.rowconfigure(0, weight=1)

        # 左右のペインを完全に均等なサイズで配置
        self.pane_l = VideoPlayerPane(viewer_frame, "左", self)
        self.pane_l.grid(row=0, column=0, sticky="nsew", padx=5)
        
        self.pane_r = VideoPlayerPane(viewer_frame, "右", self)
        self.pane_r.grid(row=0, column=1, sticky="nsew", padx=5)

        # 下部コントロールパネル
        bottom_frame = tk.Frame(self)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)

        center_container = tk.Frame(bottom_frame)
        center_container.pack(anchor=tk.CENTER)

        # マーカーボックス
        marker_frame = tk.LabelFrame(center_container, text="フレームマーカー (全5個)", font=("Meiryo", 9))
        marker_frame.pack(side=tk.LEFT, padx=15, anchor=tk.N)
        self.marker_labels = []
        
        for i in range(5):
            m_row = tk.Frame(marker_frame)
            m_row.pack(fill=tk.X, pady=1)
            
            idx = i
            tk.Button(m_row, text=f"M{i+1}保存", width=8, font=("Meiryo", 9), command=lambda ix=idx: self.save_marker(ix)).pack(side=tk.LEFT, padx=1)
            tk.Button(m_row, text="読込", width=5, font=("Meiryo", 9), command=lambda ix=idx: self.load_marker(ix)).pack(side=tk.LEFT, padx=1)
            tk.Button(m_row, text="クリア", width=5, font=("Meiryo", 9), command=lambda ix=idx: self.clear_marker(ix)).pack(side=tk.LEFT, padx=1)
            
            lbl = tk.Label(m_row, text="未設定", width=22, anchor="w", font=("Meiryo", 9))
            lbl.pack(side=tk.LEFT, padx=5)
            self.marker_labels.append(lbl)

        # アクションボックス
        action_frame = tk.Frame(center_container)
        action_frame.pack(side=tk.LEFT, padx=15, anchor=tk.N)

        tk.Button(action_frame, text="左右入れ替え", font=("Meiryo", 9), command=self.swap_panes).pack(fill=tk.X, pady=2)
        
        self.btn_sync_play = tk.Button(action_frame, text="同時 再生/一時停止", font=("Meiryo", 9), command=self.toggle_sync_play)
        self.btn_sync_play.pack(fill=tk.X, pady=2)

        sync_sub = tk.Frame(action_frame)
        sync_sub.pack(fill=tk.X, pady=2)
        tk.Button(sync_sub, text="[<] 同時 コマ戻し", font=("Meiryo", 9), command=self.sync_prev).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(sync_sub, text="同時 コマ送り [>]", font=("Meiryo", 9), command=self.sync_next).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        rot_sub = tk.Frame(action_frame)
        rot_sub.pack(fill=tk.X, pady=2)
        tk.Button(rot_sub, text="? 左回転", font=("Meiryo", 9), command=lambda: [self.pane_l.rotate(False), self.pane_r.rotate(False)]).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        tk.Button(rot_sub, text="右回転 ?", font=("Meiryo", 9), command=lambda: [self.pane_l.rotate(True), self.pane_r.rotate(True)]).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        tk.Button(action_frame, text="左右同時 画像保存", font=("Meiryo", 9), command=self.save_snapshots).pack(fill=tk.X, pady=2)

        self.lbl_diff = tk.Label(action_frame, text="時間差: 0ms (フレーム差: 0)", width=32, font=("Meiryo", 10, "bold"))
        self.lbl_diff.pack(pady=5)

        # ステータスバー（クレジット情報を含めた表示）
        status_bar = tk.Label(self, text=f"JTwinVideoDuet2Pic v{MainApp.VERSION} | Libraries: OpenCV, Pillow | Copyright c 2026 ranorat", anchor="e", fg="gray", font=("Meiryo", 8))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5)

    def on_window_resize(self, event):
        if event.widget == self:
            self.pane_l.display_frame()
            self.pane_r.display_frame()

    def update_diff_display(self):
        f_l = self.pane_l.current_frame_idx
        f_r = self.pane_r.current_frame_idx
        diff = f_l - f_r
        fps = self.pane_l.fps if self.pane_l.fps > 0 else 30.0
        ms_diff = int((diff / fps) * 1000)
        self.lbl_diff.config(text=f"時間差: {ms_diff} ms (フレーム差: L {'+' if diff >= 0 else ''}{diff})")

    def toggle_sync_play(self):
        if self.is_playing:
            self.stop_sync_play()
        else:
            self.start_sync_play()

    def start_sync_play(self):
        if self.is_playing:
            return
        
        self.pane_l.stop_play()
        self.pane_r.stop_play()

        max_l = self.pane_l.total_frames
        max_r = self.pane_r.total_frames
        if (max_l > 0 and self.pane_l.current_frame_idx >= max_l - 1) or \
           (max_r > 0 and self.pane_r.current_frame_idx >= max_r - 1):
            return

        self.is_playing = True
        self.play_loop()

    def stop_sync_play(self):
        self.is_playing = False
        if self.play_job is not None:
            self.after_cancel(self.play_job)
            self.play_job = None

    def play_loop(self):
        if not self.is_playing:
            return
            
        max_l = self.pane_l.total_frames
        max_r = self.pane_r.total_frames
        
        next_l = self.pane_l.current_frame_idx + 1
        next_r = self.pane_r.current_frame_idx + 1
        
        if (max_l > 0 and next_l >= max_l) or (max_r > 0 and next_r >= max_r):
            if next_l < max_l:
                self.pane_l.seek_frame(next_l)
            if next_r < max_r:
                self.pane_r.seek_frame(next_r)
            self.stop_sync_play()
            return
            
        if self.pane_l.cap and max_l > 0:
            self.pane_l.seek_frame(next_l)
        if self.pane_r.cap and max_r > 0:
            self.pane_r.seek_frame(next_r)
            
        fps = self.pane_l.fps if (self.pane_l.cap and self.pane_l.fps > 0) else 30.0
        interval = int(1000.0 / fps)
        self.play_job = self.after(interval, self.play_loop)

    def sync_prev(self):
        self.stop_sync_play()
        self.pane_l.step_prev()
        self.pane_r.step_prev()

    def sync_next(self):
        self.stop_sync_play()
        max_l = self.pane_l.total_frames
        max_r = self.pane_r.total_frames
        
        if (max_l > 0 and self.pane_l.current_frame_idx >= max_l - 1) or \
           (max_r > 0 and self.pane_r.current_frame_idx >= max_r - 1):
            return
            
        self.pane_l.step_next()
        self.pane_r.step_next()

    def save_marker(self, idx):
        self.markers[idx].frame_l = self.pane_l.current_frame_idx
        self.markers[idx].frame_r = self.pane_r.current_frame_idx
        self.markers[idx].is_saved = True
        self.marker_labels[idx].config(text=f"L:f{self.markers[idx].frame_l} / R:f{self.markers[idx].frame_r}")

    def load_marker(self, idx):
        if self.markers[idx].is_saved:
            self.stop_sync_play()
            self.pane_l.stop_play()
            self.pane_r.stop_play()
            self.pane_l.seek_frame(self.markers[idx].frame_l)
            self.pane_r.seek_frame(self.markers[idx].frame_r)

    def clear_marker(self, idx):
        self.markers[idx].is_saved = False
        self.marker_labels[idx].config(text="未設定")

    def swap_panes(self):
        path_l, idx_l = self.pane_l.file_path, self.pane_l.current_frame_idx
        path_r, idx_r = self.pane_r.file_path, self.pane_r.current_frame_idx
        
        for m in self.markers:
            if m.is_saved:
                m.frame_l, m.frame_r = m.frame_r, m.frame_l

        for i, m in enumerate(self.markers):
            if m.is_saved:
                self.marker_labels[i].config(text=f"L:f{m.frame_l} / R:f{m.frame_r}")
            else:
                self.marker_labels[i].config(text="未設定")

        if path_r:
            self.pane_l.load_file(path_r, is_sub_call=True)
            self.pane_l.seek_frame(idx_r)
        if path_l:
            self.pane_r.load_file(path_l, is_sub_call=True)
            self.pane_r.seek_frame(idx_l)

    def save_snapshots(self):
        if not self.pane_l.cap and not self.pane_r.cap:
            return
            
        init_file = f"{self.last_saved_filename}.png" if self.last_saved_filename else ""

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG画像", "*.png")],
            initialfile=init_file,
            title="左右の画像を保存"
        )
        if not file_path:
            return
            
        dir_name = os.path.dirname(file_path)
        raw_base_name = os.path.basename(file_path)
        base_name, _ = os.path.splitext(raw_base_name)
        
        self.last_saved_filename = base_name
        
        name_l = self.pane_l.file_basename if self.pane_l.file_basename else "left"
        name_r = self.pane_r.file_basename if self.pane_r.file_basename else "right"
        
        # 保存用ヘルパー関数：現在の回転角度を適用する
        def apply_rotation(frame, angle):
            if angle == 90:
                return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                return cv2.rotate(frame, cv2.ROTATE_180)
            elif angle == 270:
                return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return frame

        # 左側保存
        if self.pane_l.cap:
            self.pane_l.cap.set(cv2.CAP_PROP_POS_FRAMES, self.pane_l.current_frame_idx)
            ret, frame = self.pane_l.cap.read()
            if ret:
                save_path_l = os.path.join(dir_name, f"{base_name}_{name_l}[{self.pane_l.current_frame_idx}].png")
                # 回転適用
                rotated_frame = apply_rotation(frame, self.pane_l.current_angle)
                frame_rgb = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                img_pil.save(save_path_l)
                
        # 右側保存
        if self.pane_r.cap:
            self.pane_r.cap.set(cv2.CAP_PROP_POS_FRAMES, self.pane_r.current_frame_idx)
            ret, frame = self.pane_r.cap.read()
            if ret:
                save_path_r = os.path.join(dir_name, f"{base_name}_{name_r}[{self.pane_r.current_frame_idx}].png")
                # 回転適用
                rotated_frame = apply_rotation(frame, self.pane_r.current_angle)
                frame_rgb = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                img_pil.save(save_path_r)


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
