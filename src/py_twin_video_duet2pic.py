import os
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

def calculate_three_sbs_affinities(frame_l, frame_r, baseline_cm=6.5):
    """3つのシチュエーションごとのSBS親密度(0-100%)を同時に計算する"""
    img_height, img_width = frame_l.shape[:2]

    ratio = baseline_cm / 6.5
    max_y_disp = img_height * 0.02
    max_x_disp = img_width * 0.05 * ratio

    gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1500)
    kp_l, des_l = orb.detectAndCompute(gray_l, None)
    kp_r, des_r = orb.detectAndCompute(gray_r, None)

    scores = {"horizontal": 0.0, "camera_rot": 0.0, "obj_rot": 0.0}
    if des_l is None or des_r is None or len(des_l) < 10 or len(des_r) < 10:
        return scores

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_l, des_r)
    if not matches:
        return scores

    valid_matches = []
    for m in matches:
        if m.distance > 40:
            continue
        pt_l = kp_l[m.queryIdx].pt
        pt_r = kp_r[m.trainIdx].pt

        diff_y = abs(pt_l[1] - pt_r[1])
        diff_x = pt_l[0] - pt_r[0]

        if diff_y < max_y_disp and abs(diff_x) < max_x_disp:
            valid_matches.append({"diff_x": diff_x, "pt_l": pt_l, "pt_r": pt_r})

    if not valid_matches:
        return scores

    diffs_x = [m["diff_x"] for m in valid_matches]
    total_matches_count = len(matches)

    std_x = np.std(diffs_x)
    mean_x = np.mean(diffs_x)
    
    if std_x < (img_width * 0.01) and abs(mean_x) > (img_width * 0.002):
        scores["horizontal"] = (len(valid_matches) / total_matches_count) * 100.0
    else:
        penalty = max(0.0, 1.0 - (std_x / (img_width * 0.03)))
        scores["horizontal"] = (len(valid_matches) / total_matches_count) * 100.0 * penalty

    has_depth_variation = (np.min(diffs_x) < 0) and (np.max(diffs_x) > 0)
    if has_depth_variation:
        scores["camera_rot"] = (len(valid_matches) / total_matches_count) * 100.0
    else:
        scores["camera_rot"] = (len(valid_matches) / total_matches_count) * 100.0 * 0.5

    background_matches = sum(1 for dx in diffs_x if abs(dx) < (img_width * 0.002))
    object_matches = len(valid_matches) - background_matches

    if background_matches > 5 and object_matches > 5:
        scores["obj_rot"] = (len(valid_matches) / total_matches_count) * 100.0
    else:
        scores["obj_rot"] = (len(valid_matches) / total_matches_count) * 100.0 * 0.4

    for k in scores:
        scores[k] = round(min(max(scores[k], 0.0), 100.0), 1)

    return scores


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
        
        self._prev_frame_idx_for_sync = 0
        
        canvas_container = tk.Frame(self, bg="black")
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        canvas_container.pack_propagate(False)

        self.canvas_label = tk.Label(canvas_container, bg="black", anchor="center")
        self.canvas_label.pack(fill=tk.BOTH, expand=True)
        
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
        
        self.slider = tk.Scale(self, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_move)
        self.slider.pack(fill=tk.X, pady=2)
        self.slider.bind("<ButtonPress-1>", self.on_slider_press)
        self.slider.bind("<ButtonRelease-1>", self.on_slider_release)
        
        self.lbl_time = tk.Label(self, text="00:00 (Frame: 0)", font=("Meiryo", 9))
        self.lbl_time.pack(pady=2)

    def load_file(self, filepath=None, is_sub_call=False):
        self.stop_play()
        if not filepath:
            filepath = filedialog.askopenfilename(title=f"{self.title_prefix}の動画を選択", filetypes=[("動画", "*.mp4 *.ts *.mpg *.mpeg")])
        if not filepath: return
            
        if not is_sub_call:
            self.app_ref.sync_slider_var.set(False)
            for m in self.app_ref.markers: m.is_saved = False
            for lbl in self.app_ref.marker_labels: lbl.config(text="未設定")

        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(filepath)
        if not self.cap.isOpened():
            if not is_sub_call: messagebox.showerror("エラー", "動画を開けませんでした。")
            return
            
        self.file_path = filepath
        self.file_basename = os.path.splitext(os.path.basename(filepath))[0]
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            
        self.slider.config(to=max(0, self.total_frames - 1))
        self.current_frame_idx = 0
        self._prev_frame_idx_for_sync = 0
        self.seek_frame(0)
        self.app_ref.update_diff_display()

        if not is_sub_call:
            other = self.app_ref.pane_r if self.title_prefix == "左" else self.app_ref.pane_l
            if other.cap is None and messagebox.askyesno("確認", "もう一方にも同じ動画を読み込みますか？"):
                other.load_file(filepath, is_sub_call=True)

    def get_max_movable_range(self, req_idx):
        if not self.app_ref.sync_slider_var.get(): return req_idx
        other = self.app_ref.pane_r if self.title_prefix == "左" else self.app_ref.pane_l
        if not other.cap: return req_idx
        diff = self.app_ref.pane_l.current_frame_idx - self.app_ref.pane_r.current_frame_idx
        if self.title_prefix == "左":
            tr = req_idx - diff
            if tr < 0: return diff
            if tr >= other.total_frames: return other.total_frames - 1 + diff
        else:
            tl = req_idx + diff
            if tl < 0: return -diff
            if tl >= other.total_frames: return other.total_frames - 1 - diff
        return req_idx

    def seek_frame(self, frame_idx, from_sync=False):
        if not self.cap: return
        
        if not from_sync and self.app_ref.sync_slider_var.get():
            frame_idx = self.get_max_movable_range(frame_idx)

        self.current_frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret: self.display_frame(frame)
        
        if not self.is_dragging: self.slider.set(self.current_frame_idx)
        self.update_time_label()

        if not from_sync and self.app_ref.sync_slider_var.get():
            other = self.app_ref.pane_r if self.title_prefix == "左" else self.app_ref.pane_l
            if other.cap:
                delta = self.current_frame_idx - self._prev_frame_idx_for_sync
                target_other_idx = other.current_frame_idx + delta
                other.seek_frame(target_other_idx, from_sync=True)
                
        self._prev_frame_idx_for_sync = self.current_frame_idx


    def display_frame(self, frame=None):
        if frame is not None: 
            self.last_frame = frame
        elif self.last_frame is None: 
            return
            
        ft = self.last_frame.copy()

        is_any_play = False
        if hasattr(self, 'app_ref') and self.app_ref:
            is_any_play = self.app_ref.is_any_playing()
        else:
            is_any_play = self.is_playing

        if getattr(self, 'show_grid', False) and not is_any_play:
            h, w = ft.shape[:2]
            color = (0, 255, 128)
            thickness = 1

            for x_pos in [w // 3, (w * 2) // 3]:
                cv2.line(ft, (x_pos, 0), (x_pos, h), color, thickness, cv2.LINE_AA)
            for y_pos in [h // 3, (h * 2) // 3]:
                cv2.line(ft, (0, y_pos), (w, y_pos), color, thickness, cv2.LINE_AA)
                
            cv2.line(ft, (w // 2, 0), (w // 2, h), (0, 0, 255), 1, cv2.LINE_AA)

        if self.current_angle == 90: ft = cv2.rotate(ft, cv2.ROTATE_90_CLOCKWISE)
        elif self.current_angle == 180: ft = cv2.rotate(ft, cv2.ROTATE_180)
        elif self.current_angle == 270: ft = cv2.rotate(ft, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        img = Image.fromarray(cv2.cvtColor(ft, cv2.COLOR_BGR2RGB))
        self.update_idletasks()
        w, h = max(self.canvas_label.winfo_width(), 300), max(self.canvas_label.winfo_height(), 200)
        ratio = min(w / img.width, h / img.height)
        self.photo = ImageTk.PhotoImage(img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))), Image.Resampling.LANCZOS))
        self.canvas_label.config(image=self.photo)


    def update_time_label(self):
        m, s = divmod(int(self.current_frame_idx / self.fps if self.fps > 0 else 0), 60)
        self.lbl_time.config(text=f"{m:02d}:{s:02d} (Frame: {self.current_frame_idx})")
        self.app_ref.update_diff_display()

    def toggle_play(self):
        if self.app_ref.check_and_disable_sync("再生"):
            if self.is_playing: 
                self.stop_play()
            else:
                if self.total_frames > 0 and self.current_frame_idx >= self.total_frames - 1: return
                self.app_ref.stop_sync_play()
                self.is_playing = True
                
                if self.app_ref.pane_l: self.app_ref.pane_l.refresh_display()
                if self.app_ref.pane_r: self.app_ref.pane_r.refresh_display()
                
                self.play_loop()

    def step_next(self):
        if self.app_ref.check_and_disable_sync("コマ送り"):
            if self.current_frame_idx < self.total_frames - 1: 
                self.seek_frame(self.current_frame_idx + 1)
                self.app_ref.run_sbs_analysis()

    def step_prev(self):
        if self.app_ref.check_and_disable_sync("コマ戻し"):
            self.stop_play()
            if self.current_frame_idx > 0: 
                self.seek_frame(self.current_frame_idx - 1)
                self.app_ref.run_sbs_analysis()

    def stop_play(self):
        self.is_playing = False
        if self.play_job: 
            self.after_cancel(self.play_job)
            self.play_job = None
            
        if self.app_ref and not self.app_ref.is_any_playing():
            if self.app_ref.pane_l: self.app_ref.pane_l.refresh_display()
            if self.app_ref.pane_r: self.app_ref.pane_r.refresh_display()
            self.app_ref.run_sbs_analysis()

    def play_loop(self):
        if not self.is_playing: return
        if self.current_frame_idx < self.total_frames - 1:
            next_idx = min(self.current_frame_idx + 1, self.total_frames - 1)
            self.seek_frame(next_idx)
            self.play_job = self.after(int(1000.0 / self.fps), self.play_loop)
        else: self.stop_play()

    def on_slider_press(self, e): self.is_dragging = True

    def on_slider_release(self, e): 
        self.is_dragging = False
        self.seek_frame(int(self.slider.get()))
        self.app_ref.run_sbs_analysis()

    def on_slider_move(self, v): 
        if self.is_dragging: self.seek_frame(int(float(v)))

    def rotate(self, cw): 
        self.current_angle = (self.current_angle + (90 if cw else -90)) % 360
        self.seek_frame(self.current_frame_idx)
        self.app_ref.run_sbs_analysis()

    def refresh_display(self):
        if self.cap and self.total_frames > 0:
            self.seek_frame(self.current_frame_idx)


class MainApp(tk.Tk):
    VERSION = "1.0.2"

    def __init__(self):
        super().__init__()
        self.title("ツインビデオデュエット2Pic (Python版)")
        self.geometry("1300x820")
        self.minsize(1150, 760)
        self.bind("<Configure>", self.on_window_resize)

        self.markers = [MarkerData() for _ in range(5)]
        self.is_playing = False
        self.play_job = None
        self.last_saved_filename = ""

        menubar = tk.Menu(self)
        help_menu = tk.Menu(menubar, tearoff=0)
        about_text = (
            f"ツインビデオデュエット2Pic v{MainApp.VERSION}\n\n"
            "【サードパーティ・ライセンス / クレジット】\n"
            "・OpenCV (opencv-python)\n"
            "  Copyright (C) 2000-2008, Intel Corporation, all rights reserved.\n"
            "  Copyright (C) 2008-2009, Willow Garage Inc., all rights reserved.\n"
            "  Third-party copyrights are property of their respective owners.\n"
            "  (BSD License / Apache License 2.0)\n\n"
            "・Pillow (Python Imaging Library)\n"
            "  Copyright (c) 1997-2011 by Secret Labs AB\n"
            "  Copyright (c) 1995-2011 by Fredrik Lundh and contributors\n"
            "  Copyright (c) 2010 by Jeffrey 'Alex' Clark and contributors\n"
            "  (MIT-CMU License)\n\n"
            "Copyright c 2026 ranorat"
        )
        help_menu.add_command(label="バージョン情報", command=lambda: messagebox.showinfo("バージョン情報", about_text))
        menubar.add_cascade(label="ヘルプ", menu=help_menu)
        self.config(menu=menubar)

        viewer_frame = tk.Frame(self)
        viewer_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        viewer_frame.columnconfigure(0, weight=1, uniform="pane")
        viewer_frame.columnconfigure(1, weight=1, uniform="pane")
        viewer_frame.rowconfigure(0, weight=1)

        self.pane_l = VideoPlayerPane(viewer_frame, "左", self)
        self.pane_l.grid(row=0, column=0, sticky="nsew", padx=5)
        self.pane_r = VideoPlayerPane(viewer_frame, "右", self)
        self.pane_r.grid(row=0, column=1, sticky="nsew", padx=5)

        bottom_frame = tk.Frame(self)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)

        center_container = tk.Frame(bottom_frame)
        center_container.pack(anchor=tk.CENTER)

        # 1. フレームマーカー群（左）
        marker_frame = tk.LabelFrame(center_container, text="フレームマーカー (全5個)", font=("Meiryo", 9))
        marker_frame.pack(side=tk.LEFT, padx=10, anchor=tk.N)
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

        # 2. 同時操作・アクションボタン群（中央）
        action_frame = tk.Frame(center_container)
        action_frame.pack(side=tk.LEFT, padx=10, anchor=tk.N)

        tk.Button(action_frame, text="左右入れ替え", font=("Meiryo", 9), command=self.swap_panes).pack(fill=tk.X, pady=2)

        self.sync_slider_var = tk.BooleanVar(value=False)
        
        toggle_row = tk.Frame(action_frame)
        toggle_row.pack(fill=tk.X, pady=2)
        
        self.chk_sync = tk.Checkbutton(toggle_row, text="スライダー同期", variable=self.sync_slider_var,
                                       command=self.on_toggle_sync_slider, font=("Meiryo", 9))
        self.chk_sync.pack(side=tk.LEFT, expand=True, anchor="w")

        self.grid_var = tk.BooleanVar(value=False)
        self.chk_grid = tk.Checkbutton(toggle_row, text="格子線表示", variable=self.grid_var,
                                     command=self.on_toggle_grid, font=("Meiryo", 9))
        self.chk_grid.pack(side=tk.LEFT, expand=True, anchor="w")

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

        # 3. SBS評価 & フレーム差分情報群（右側）
        right_info_frame = tk.Frame(center_container)
        right_info_frame.pack(side=tk.LEFT, padx=10, anchor=tk.N)

        sbs_group_frame = tk.LabelFrame(right_info_frame, text="SBS評価設定・スコア", font=("Meiryo", 9))
        sbs_group_frame.pack(fill=tk.X, pady=2)

        # --- SBS評価の有効/無効切り替えチェックボックス ---
        self.sbs_enabled_var = tk.BooleanVar(value=False)
        self.chk_sbs_enabled = tk.Checkbutton(sbs_group_frame, text="SBS評価を有効にする", variable=self.sbs_enabled_var,
                                              command=self.on_toggle_sbs_enabled, font=("Meiryo", 9))
        self.chk_sbs_enabled.pack(anchor="w", padx=5, pady=2)

        # 視差ベースライン入力行（ここにリセットボタンを1つまとめる）
        base_row = tk.Frame(sbs_group_frame)
        base_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(base_row, text="視差:", font=("Meiryo", 8)).pack(side=tk.LEFT)

        self.baseline_var = tk.StringVar(value="6.5")
        self.entry_baseline = tk.Entry(base_row, textvariable=self.baseline_var, width=5, font=("Meiryo", 9))
        self.entry_baseline.pack(side=tk.LEFT, padx=2)
        tk.Label(base_row, text="cm (1-50)", font=("Meiryo", 8)).pack(side=tk.LEFT)
        
        # 共通のリセットボタン（R）
        self.btn_reset_base = tk.Button(base_row, text="R", width=2, font=("Meiryo", 8), 
                                        command=lambda: [self.baseline_var.set("6.5"), self.run_sbs_analysis()])
        self.btn_reset_base.pack(side=tk.LEFT, padx=2)

        self.entry_baseline.bind("<FocusOut>", self.on_baseline_focus_out)

        # 3つのSBS評価（スコア表示 ＋ 抽出ボタン ※リセットは撤去してすっきり）
        self.lbl_sbs_scores = []
        keys = ["horizontal", "camera_rot", "obj_rot"]
        labels = ["①水平移動", "②カメラ回転", "③物体回転"]
        
        for i, key in enumerate(keys):
            row = tk.Frame(sbs_group_frame)
            row.pack(fill=tk.X, padx=5, pady=2)
            
            lbl = tk.Label(row, text=f"{labels[i]}: 0%", width=16, anchor="w", font=("Meiryo", 9), fg="white", bg="black")
            lbl.pack(side=tk.LEFT, padx=2)
            self.lbl_sbs_scores.append(lbl)
            
            btn_ext = tk.Button(row, text="抽出", width=5, font=("Meiryo", 8), 
                                command=lambda k=key: self.extract_optimal_parallax(k))
            btn_ext.pack(side=tk.LEFT, padx=2)

        self.lbl_diff = tk.Label(right_info_frame, text="時間差: 0ms (フレーム差: 0)", width=28, anchor="w", font=("Meiryo", 9, "bold"))
        self.lbl_diff.pack(pady=5)

        status_bar = tk.Label(self, text=f"py_twin_video_duet2pic v{MainApp.VERSION} | c 2026 ranorat", anchor="e", fg="gray", font=("Meiryo", 8))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5)

    def on_toggle_sync_slider(self):
        if self.sync_slider_var.get():
            self.stop_sync_play()
            self.pane_l.stop_play()
            self.pane_r.stop_play()

    def on_toggle_sbs_enabled(self):
        """SBS評価の有効/無効が切り替わったときの処理"""
        if self.sbs_enabled_var.get():
            self.run_sbs_analysis()
        else:
            self.reset_sbs_scores()

    def check_and_disable_sync(self, action_name):
        if self.sync_slider_var.get():
            if messagebox.askyesno("確認", f"{action_name}を実行します。スライダー同期のチェックを外しますか？"):
                self.sync_slider_var.set(False)
                return True
            return False
        return True

    def on_window_resize(self, event):
        if event.widget == self:
            self.pane_l.display_frame()
            self.pane_r.display_frame()

    def update_diff_display(self):
        diff = self.pane_l.current_frame_idx - self.pane_r.current_frame_idx
        fps = self.pane_l.fps if self.pane_l.fps > 0 else 30.0
        self.lbl_diff.config(text=f"時間差: {int((diff / fps) * 1000)} ms (フレーム差: L {'+' if diff >= 0 else ''}{diff})")

    def toggle_sync_play(self):
        if not self.check_and_disable_sync("再生"): return
        if self.is_playing: 
            self.stop_sync_play()
            self.run_sbs_analysis()
        else: self.start_sync_play()

    def start_sync_play(self):
        if self.is_playing: return
        self.pane_l.stop_play()
        self.pane_r.stop_play()
        if (self.pane_l.total_frames > 0 and self.pane_l.current_frame_idx >= self.pane_l.total_frames - 1) or \
           (self.pane_r.total_frames > 0 and self.pane_r.current_frame_idx >= self.pane_r.total_frames - 1):
            return
        self.is_playing = True
        
        if self.pane_l: self.pane_l.refresh_display()
        if self.pane_r: self.pane_r.refresh_display()
        
        self.play_loop()

    def stop_sync_play(self):
        self.is_playing = False
        if self.play_job: 
            self.after_cancel(self.play_job)
            self.play_job = None
        if hasattr(self, 'btn_sync_play'):
            self.btn_sync_play.config(text="同時 再生/一時停止")
            
        if self.pane_l: self.pane_l.refresh_display()
        if self.pane_r: self.pane_r.refresh_display()
        self.run_sbs_analysis()

    def play_loop(self):
        if not self.is_playing: return
        nl, nr = self.pane_l.current_frame_idx + 1, self.pane_r.current_frame_idx + 1
        if (self.pane_l.total_frames > 0 and nl >= self.pane_l.total_frames) or \
           (self.pane_r.total_frames > 0 and nr >= self.pane_r.total_frames):
            if nl < self.pane_l.total_frames: self.pane_l.seek_frame(nl)
            if nr < self.pane_r.total_frames: self.pane_r.seek_frame(nr)
            self.stop_sync_play()
            return
        if self.pane_l.cap: self.pane_l.seek_frame(nl)
        if self.pane_r.cap: self.pane_r.seek_frame(nr)
        fps = self.pane_l.fps if (self.pane_l.cap and self.pane_l.fps > 0) else 30.0

        self.play_job = self.after(int(1000.0 / fps), self.play_loop)

    def sync_prev(self):
        self.stop_sync_play()
        self.pane_l.stop_play()
        self.pane_r.stop_play()
        
        if self.pane_l.current_frame_idx > 0 and self.pane_r.current_frame_idx > 0:
            if self.sync_slider_var.get():
                # 同期スライダーONのときは従来通りの処理
                self.pane_l.seek_frame(self.pane_l.current_frame_idx - 1)
            else:
                # 【同時バッチ処理】描画を挟まずにファイル読み込み（seek）だけを両方行う
                target_l = max(0, self.pane_l.current_frame_idx - 1)
                target_r = max(0, self.pane_r.current_frame_idx - 1)
                
                # 左の読み込みと内部インデックス更新（描画はまだしない）
                self.pane_l.current_frame_idx = target_l
                self.pane_l.cap.set(cv2.CAP_PROP_POS_FRAMES, target_l)
                ret_l, frame_l = self.pane_l.cap.read()
                if ret_l: self.pane_l.last_frame = frame_l
                
                # 右の読み込みと内部インデックス更新（描画はまだしない）
                self.pane_r.current_frame_idx = target_r
                self.pane_r.cap.set(cv2.CAP_PROP_POS_FRAMES, target_r)
                ret_r, frame_r = self.pane_r.cap.read()
                if ret_r: self.pane_r.last_frame = frame_r
                
                # まとめてUIやスライダー、時間表示を更新
                self.pane_l.slider.set(self.pane_l.current_frame_idx)
                self.pane_r.slider.set(self.pane_r.current_frame_idx)
                self.pane_l.update_time_label() # これの中でdiffも更新される
                
                # 最後に1回ずつ描画をまとめて実行
                self.pane_l.display_frame()
                self.pane_r.display_frame()

            self.run_sbs_analysis()

    def sync_next(self):
        self.stop_sync_play()
        self.pane_l.stop_play()
        self.pane_r.stop_play()
        
        max_l = self.pane_l.total_frames - 1
        max_r = self.pane_r.total_frames - 1
        
        if self.pane_l.current_frame_idx < max_l and self.pane_r.current_frame_idx < max_r:
            if self.sync_slider_var.get():
                # 同期スライダーONのときは従来通りの処理
                self.pane_l.seek_frame(self.pane_l.current_frame_idx + 1)
            else:
                # 【同時バッチ処理】描画を挟まずにファイル読み込み（seek）だけを両方行う
                target_l = min(max_l, self.pane_l.current_frame_idx + 1)
                target_r = min(max_r, self.pane_r.current_frame_idx + 1)
                
                # 左の読み込み
                self.pane_l.current_frame_idx = target_l
                self.pane_l.cap.set(cv2.CAP_PROP_POS_FRAMES, target_l)
                ret_l, frame_l = self.pane_l.cap.read()
                if ret_l: self.pane_l.last_frame = frame_l
                
                # 右の読み込み
                self.pane_r.current_frame_idx = target_r
                self.pane_r.cap.set(cv2.CAP_PROP_POS_FRAMES, target_r)
                ret_r, frame_r = self.pane_r.cap.read()
                if ret_r: self.pane_r.last_frame = frame_r
                
                # まとめてUI更新
                self.pane_l.slider.set(self.pane_l.current_frame_idx)
                self.pane_r.slider.set(self.pane_r.current_frame_idx)
                self.pane_l.update_time_label()
                
                # 最後に1回ずつ描画
                self.pane_l.display_frame()
                self.pane_r.display_frame()

            self.run_sbs_analysis()

    def save_marker(self, idx):
        self.markers[idx].frame_l = self.pane_l.current_frame_idx
        self.markers[idx].frame_r = self.pane_r.current_frame_idx
        self.markers[idx].is_saved = True
        self.marker_labels[idx].config(text=f"L:f{self.markers[idx].frame_l} / R:f{self.markers[idx].frame_r}")

    def load_marker(self, idx):
        if self.markers[idx].is_saved:
            if not self.check_and_disable_sync("フレームマーカー読込"): 
                return
            self.stop_sync_play()
            self.pane_l.stop_play()
            self.pane_r.stop_play()
            self.pane_l.seek_frame(self.markers[idx].frame_l)
            self.pane_r.seek_frame(self.markers[idx].frame_r)
            self.run_sbs_analysis()

    def clear_marker(self, idx):
        self.markers[idx].is_saved = False
        self.marker_labels[idx].config(text="未設定")

    def swap_panes(self):
        if self.sync_slider_var.get():
            if not messagebox.askyesno("確認", "左右入れ替えを実行します。スライダー同期のチェックを外しますか？"):
                return
            self.sync_slider_var.set(False)

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
        if not self.pane_l.cap and not self.pane_r.cap: return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG画像", "*.png")],
            initialfile=f"{self.last_saved_filename}.png" if self.last_saved_filename else "",
            title="左右の画像を保存"
        )
        if not file_path: return
        dir_name, raw_base = os.path.dirname(file_path), os.path.basename(file_path)
        base_name, _ = os.path.splitext(raw_base)
        self.last_saved_filename = base_name
        
        nl = self.pane_l.file_basename or "left"
        nr = self.pane_r.file_basename or "right"
        
        def rot(f, a):
            return cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE) if a == 90 else (cv2.rotate(f, cv2.ROTATE_180) if a == 180 else (cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE) if a == 270 else f))

        for pane, name in [(self.pane_l, nl), (self.pane_r, nr)]:
            if pane.cap:
                pane.cap.set(cv2.CAP_PROP_POS_FRAMES, pane.current_frame_idx)
                ret, frame = pane.cap.read()
                if ret:
                    Image.fromarray(cv2.cvtColor(rot(frame, pane.current_angle), cv2.COLOR_BGR2RGB)).save(
                        os.path.join(dir_name, f"{base_name}_{name}[{pane.current_frame_idx}].png")
                    )

    def run_sbs_analysis(self):
        """【高速化＆有効/無効チェック対応】軽量解像度でSBS分析を実行する"""
        # チェックがOFFならスコアを0にして即終了
        if not self.sbs_enabled_var.get():
            self.reset_sbs_scores()
            return

        if (self.pane_l and self.pane_l.is_playing) or \
           (self.pane_r and self.pane_r.is_playing) or \
           (hasattr(self, 'is_playing') and self.is_playing):
            self.reset_sbs_scores()
            return
            
        if self.pane_l.last_frame is None or self.pane_r.last_frame is None:
            return

        val_str = self.baseline_var.get().strip()
        baseline_val = self.clamp_and_format_parallax(val_str)
        self.baseline_var.set(str(baseline_val))

        # 回転後のフレーム（格子線なし）
        frame_l_processed = self.get_rotated_frame(self.pane_l.last_frame, self.pane_l.current_angle)
        frame_r_processed = self.get_rotated_frame(self.pane_r.last_frame, self.pane_r.current_angle)

        # 【超高速化】フルHD等の高解像度対策として、分析用画像を幅最大800pxに縮小して処理（格子線が混入する心配もナシ）
        h, w = frame_l_processed.shape[:2]
        max_analysis_width = 800
        if w > max_analysis_width:
            scale = max_analysis_width / w
            new_w = max_analysis_width
            new_h = int(h * scale)
            frame_l_processed = cv2.resize(frame_l_processed, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frame_r_processed = cv2.resize(frame_r_processed, (new_w, new_h), interpolation=cv2.INTER_AREA)

        scores = calculate_three_sbs_affinities(frame_l_processed, frame_r_processed, baseline_cm=baseline_val)

        keys = ["horizontal", "camera_rot", "obj_rot"]
        labels = ["①水平移動", "②カメラ回転", "③物体回転"]
        for i, key in enumerate(keys):
            score = scores[key]
            color = "red" if score >= 70.0 else ("yellow" if score >= 40.0 else "white")
            self.lbl_sbs_scores[i].config(text=f"{labels[i]}: {score}%", fg=color)

    def reset_sbs_scores(self):
        labels = ["①水平移動", "②カメラ回転", "③物体回転"]
        for i in range(3):
            self.lbl_sbs_scores[i].config(text=f"{labels[i]}: 0.0%", fg="white")

    def get_rotated_frame(self, frame, angle):
        if angle == 90: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180: return cv2.rotate(frame, cv2.ROTATE_180)
        if angle == 270: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame.copy()

    def clamp_and_format_parallax(self, val_str):
        try:
            val = float(val_str)
        except (ValueError, TypeError):
            return 6.5

        if val < 1.0:
            val = 1.0
        elif val > 50.0:
            val = 50.0
        return round(val, 1)

    def on_baseline_focus_out(self, event):
        val_str = self.baseline_var.get().strip()
        if not val_str:
            self.baseline_var.set("6.5")
            return
        clamped = self.clamp_and_format_parallax(val_str)
        self.baseline_var.set(str(clamped))
        self.run_sbs_analysis()

    def extract_optimal_parallax(self, target_key):
        if not self.sbs_enabled_var.get():
            messagebox.showinfo("情報", "「SBS評価を有効にする」にチェックが入っていません。")
            return

        if (self.pane_l and self.pane_l.is_playing) or \
           (self.pane_r and self.pane_r.is_playing) or \
           (hasattr(self, 'is_playing') and self.is_playing):
            return
        if self.pane_l.last_frame is None or self.pane_r.last_frame is None:
            return

        frame_l_processed = self.get_rotated_frame(self.pane_l.last_frame, self.pane_l.current_angle)
        frame_r_processed = self.get_rotated_frame(self.pane_r.last_frame, self.pane_r.current_angle)

        h, w = frame_l_processed.shape[:2]
        max_analysis_width = 800
        if w > max_analysis_width:
            scale = max_analysis_width / w
            new_w = max_analysis_width
            new_h = int(h * scale)
            frame_l_processed = cv2.resize(frame_l_processed, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frame_r_processed = cv2.resize(frame_r_processed, (new_w, new_h), interpolation=cv2.INTER_AREA)

        best_parallax = 6.5
        max_score = -1.0

        for p in np.arange(1.0, 50.1, 0.5):
            p_val = round(float(p), 1)
            scores = calculate_three_sbs_affinities(frame_l_processed, frame_r_processed, baseline_cm=p_val)
            current_score = scores[target_key]
            
            if current_score > max_score:
                max_score = current_score
                best_parallax = p_val

        final_val = self.clamp_and_format_parallax(str(best_parallax))
        self.baseline_var.set(str(final_val))
        self.run_sbs_analysis()

    def on_toggle_grid(self):
        show_grid = self.grid_var.get()
        if self.pane_l:
            self.pane_l.show_grid = show_grid
            self.pane_l.refresh_display()
        if self.pane_r:
            self.pane_r.show_grid = show_grid
            self.pane_r.refresh_display()

    def is_any_playing(self):
        pane_l_playing = self.pane_l and self.pane_l.is_playing
        pane_r_playing = self.pane_r and self.pane_r.is_playing
        return self.is_playing or pane_l_playing or pane_r_playing


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
