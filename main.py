import os
import cv2
import librosa
import numpy as np
import subprocess
import json
import threading
import hashlib
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from concurrent.futures import ProcessPoolExecutor, as_completed

CACHE_FILE = "analysis_cache.json"
cancel_processing = False  # للتحكم في الإلغاء
AUDIO_ANALYSIS_SECONDS = 90  # مدة جزئية لتحليل الإيقاع فقط

# تحميل الكاش لو موجود
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        analysis_cache = json.load(f)
else:
    analysis_cache = {"videos": {}, "audios": {}}

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(analysis_cache, f, ensure_ascii=False, indent=2)

# ===============================
# أدوات مساعدة
# ===============================

def file_md5(path, chunk_size=2**20):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()

def probe_duration(path):
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ], stderr=subprocess.STDOUT).decode().strip()
        return round(float(out), 2)
    except Exception:
        return 0.0

def has_audio_stream(path):
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-select_streams", "a", "-show_entries", "stream=index",
            "-of", "csv=p=0", path
        ], stderr=subprocess.STDOUT).decode().strip()
        return bool(out)
    except Exception:
        return False

# ===============================
# إزالة الصوت من الفيديوهات (نسخ سريع)
# ===============================

def remove_audio_from_videos(input_dir, temp_dir, progress_callback=None):
    os.makedirs(temp_dir, exist_ok=True)
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]

    for i, filename in enumerate(video_files):
        if cancel_processing:
            break
        try:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(temp_dir, filename)
            command = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-map", "0:v:0",
                "-c:v", "copy",
                "-an",
                "-movflags", "+faststart",
                output_path
            ]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            if progress_callback:
                progress_callback(i + 1, len(video_files))
        except Exception as e:
            messagebox.showerror("خطأ في إزالة الصوت", f"❌ {filename}:\n{str(e)}")

# ===============================
# تحليل الفيديو
# ===============================

def analyze_video(video_path):
    if video_path in analysis_cache["videos"]:
        return video_path, analysis_cache["videos"][video_path]["duration"], analysis_cache["videos"][video_path]["motion"]

    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps else probe_duration(video_path)
        motion_score = []
        prev_frame = None

        step = int(fps) if fps and fps > 0 else 1  # لقطة كل ثانية

        for i in range(0, int(frame_count), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                motion_score.append(np.mean(diff))
            prev_frame = gray

        cap.release()
        avg_motion = np.mean(motion_score) if motion_score else 0
        analysis_cache["videos"][video_path] = {"duration": round(duration, 2), "motion": round(avg_motion, 2)}
        save_cache()
        return video_path, round(duration, 2), round(avg_motion, 2)
    except Exception:
        return video_path, 0, 0

# ===============================
# تحليل الصوت (قبول بدون Tempo إذا تعذّر)
# ===============================

def analyze_audio(audio_path):
    # كاش
    if audio_path in analysis_cache["audios"]:
        d = analysis_cache["audios"][audio_path]["duration"]
        t = analysis_cache["audios"][audio_path].get("tempo")
        return audio_path, d, t

    duration = probe_duration(audio_path)  # نحاول ffprobe أولاً
    tempo = None

    try:
        # تحميل جزئي لاستخراج الإيقاع فقط (تخفيفًا للوقت/الذاكرة)
        y, sr = librosa.load(audio_path, mono=True, sr=22050,
                             duration=AUDIO_ANALYSIS_SECONDS, res_type="kaiser_fast")
        if not duration or duration <= 0:
            duration = librosa.get_duration(y=y, sr=sr)

        t, _ = librosa.beat.beat_track(y=y, sr=sr)
        t = float(t)
        # لو اكتشاف الإيقاع صفر (موسيقى أمبيانت)، نعدّه غير متاح
        if t > 0:
            tempo = t
    except Exception:
        # لا مشكلة: سنقبل الملف طالما المدة معروفة
        pass

    # لا نقبل إلا لو المدة معروفة وموجبة
    if not duration or duration <= 0:
        return audio_path, None, None

    # خزن في الكاش
    analysis_cache["audios"][audio_path] = {
        "duration": round(duration, 2),
        "tempo": round(tempo, 2) if tempo is not None else None
    }
    save_cache()

    return audio_path, round(duration, 2), (round(tempo, 2) if tempo is not None else None)

# ===============================
# تصنيف (غير إلزامي للربط الحالي)
# ===============================

def get_motion_category(motion):
    if motion >= 15:
        return "fast"
    elif motion >= 3:
        return "medium"
    return "slow"

def get_tempo_category(tempo):
    if tempo is None:
        return "slow"
    if tempo >= 120:
        return "fast"
    elif tempo >= 90:
        return "medium"
    return "slow"

# ===============================
# الدمج مع Fade-in/Fade-out
# ===============================

def merge_audio_video_with_fade(video_path, audio_path, output_path, video_duration):
    base, _ = os.path.splitext(output_path)
    out_mp4 = base + ".mp4"

    # حساب أزمنة الدخول/الخروج الصوتي
    if video_duration and video_duration > 4:
        fade_in_d = 2.0
        fade_out_d = 2.0
        fade_out_start = max(0.0, float(video_duration) - fade_out_d)
    else:
        fade_in_d = max(0.2, (video_duration or 4) / 8.0)
        fade_out_d = fade_in_d
        fade_out_start = max(0.0, float(video_duration or 4) - fade_out_d)

    afilter = f"afade=t=in:st=0:d={fade_in_d},afade=t=out:st={fade_out_start}:d={fade_out_d}"

    # محاولة 1: نسخ الفيديو
    cmd_copy = [
        "ffmpeg", "-y",
        "-i", video_path, "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-filter:a", afilter,
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        out_mp4
    ]
    try:
        subprocess.run(cmd_copy, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if not has_audio_stream(out_mp4):
            raise RuntimeError("no audio after copy")
        return out_mp4
    except Exception:
        # محاولة 2: إعادة ترميز الفيديو فقط عند الحاجة
        cmd_reencode = [
            "ffmpeg", "-y",
            "-i", video_path, "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-filter:a", afilter,
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            out_mp4
        ]
        subprocess.run(cmd_reencode, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if not has_audio_stream(out_mp4):
            raise RuntimeError("no audio in output")
        return out_mp4

# ===============================
# اختيار الصوت لكل فيديو (بدون فيديو صامت)
# ===============================

def assign_audios_to_videos(video_analysis, available_audios):
    """
    يضمن أن كل فيديو يملك صوتًا:
      - إذا كان عدد الأصوات >= عدد الفيديوهات: كل فيديو يحصل على صوت مختلف (فريد) حتى لو غير مطابق.
      - إذا كان عدد الفيديوهات > عدد الأصوات: يسمح بالتكرار مع توزيع متوازن.
    يتم اختيار الأقرب في المدة لتقليل الفروقات.
    """
    if not available_audios:
        for v in video_analysis:
            v["audio"] = None
        return

    for a in available_audios:
        a.setdefault("use_count", 0)
        a.setdefault("duration", float(a.get("duration") or 0.0))
    for v in video_analysis:
        v.setdefault("duration", float(v.get("duration") or 0.0))

    V = len(video_analysis)
    A = len(available_audios)

    if A >= V:
        remaining_audios = available_audios.copy()
        videos_sorted = sorted(video_analysis, key=lambda x: x["duration"], reverse=True)
        for v in videos_sorted:
            best_idx = min(range(len(remaining_audios)), key=lambda i: abs(remaining_audios[i]["duration"] - v["duration"]))
            chosen = remaining_audios.pop(best_idx)
            v["audio"] = chosen
            chosen["use_count"] += 1
        return

    # A < V -> نوزّع التكرار بالتساوي
    base = V // A
    extra = V % A
    quotas = [base + (1 if i < extra else 0) for i in range(A)]

    audios_sorted = sorted(available_audios, key=lambda x: x["duration"], reverse=True)
    remaining_videos = sorted(video_analysis, key=lambda x: x["duration"], reverse=True)

    for idx, a in enumerate(audios_sorted):
        q = quotas[idx]
        for _ in range(q):
            if not remaining_videos:
                break
            best_vid_idx = min(range(len(remaining_videos)), key=lambda i: abs(remaining_videos[i]["duration"] - a["duration"]))
            v = remaining_videos.pop(best_vid_idx)
            v["audio"] = a
            a["use_count"] += 1

    while remaining_videos:
        v = remaining_videos.pop(0)
        a = min(available_audios, key=lambda x: (x["use_count"], abs(x["duration"] - v["duration"])))
        v["audio"] = a
        a["use_count"] += 1

# ===============================
# العملية الكاملة
# ===============================

def process_all(videos_dir, audios_dir, output_dir, update_progress):
    global cancel_processing
    cancel_processing = False

    try:
        messagebox.showinfo("البدء", "🔄 جاري معالجة الفيديوهات...")
        os.makedirs(output_dir, exist_ok=True)
        temp_dir = os.path.join(videos_dir, "_no_audio_temp")
        remove_audio_from_videos(videos_dir, temp_dir)

        if cancel_processing:
            messagebox.showinfo("إلغاء", "❌ تم إلغاء العملية.")
            return

        video_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
        audio_files = [os.path.join(audios_dir, f) for f in os.listdir(audios_dir) if f.lower().endswith(('.mp3', '.wav', '.m4a', '.flac'))]

        if not audio_files:
            messagebox.showerror("لا توجد أصوات", "❌ لم يتم العثور على أي ملفات صوت. لا يمكن إنشاء فيديوهات بلا صوت.")
            return

        # ---- تحليل الأصوات بالتوازي ----
        available_audios = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(analyze_audio, audio) for audio in audio_files]
            for i, future in enumerate(as_completed(futures)):
                if cancel_processing:
                    break
                audio_path, duration, tempo = future.result()
                # ✅ نقبل الصوت إذا كانت المدة > 0 حتى لو tempo غير متاح
                if duration and duration > 0:
                    available_audios.append({
                        "file": os.path.basename(audio_path),
                        "path": audio_path,
                        "duration": duration,
                        "tempo": tempo  # قد يكون None، لا مشكلة
                    })

        if cancel_processing:
            messagebox.showinfo("إلغاء", "❌ تم إلغاء العملية.")
            return

        if not available_audios:
            messagebox.showerror("لا توجد أصوات صالحة", "❌ تعذّر تحليل أي ملف صوت (قد تكون الملفات تالفة/غير مدعومة).")
            return

        # ---- تحليل الفيديوهات بالتوازي ----
        video_analysis = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(analyze_video, video) for video in video_files]
            for i, future in enumerate(as_completed(futures)):
                if cancel_processing:
                    break
                video_path, duration, motion = future.result()
                motion_cat = get_motion_category(motion)
                video_analysis.append({
                    "file": os.path.basename(video_path),
                    "path": video_path,
                    "duration": duration,
                    "motion_cat": motion_cat,
                    "audio": None
                })
                update_progress(i + 1, len(video_files))

        if cancel_processing:
            messagebox.showinfo("إلغاء", "❌ تم إلغاء العملية.")
            return

        # ---- تعيين صوت لكل فيديو (دائمًا) ----
        assign_audios_to_videos(video_analysis, available_audios)

        # تأكيد نهائي: عيّن الأقل استخدامًا لأي فيديو بلا صوت (تحسبًا)
        for v in video_analysis:
            if not v.get("audio"):
                a = min(available_audios, key=lambda x: x["use_count"])
                v["audio"] = a
                a["use_count"] += 1

        # ---- الدمج مع Fade لكل فيديو ----
        for i, video in enumerate(video_analysis):
            if cancel_processing:
                break
            output_path = os.path.join(output_dir, video["file"])
            try:
                merge_audio_video_with_fade(video["path"], video["audio"]["path"], output_path, video["duration"])
            except Exception as e:
                # إعادة محاولة بترميز (نفس الدالة تقوم بهذا غالبًا)، وإن فشل نتخطى دون إنتاج صامت
                try:
                    merge_audio_video_with_fade(video["path"], video["audio"]["path"], output_path, video["duration"])
                except Exception as e2:
                    messagebox.showerror("خطأ دمج", f"فشل دمج {video['file']} مع {video['audio']['file']}:\n{e2}")

            update_progress(i + 1, len(video_analysis))

        if cancel_processing:
            messagebox.showinfo("إلغاء", "❌ تم إلغاء العملية.")
        else:
            messagebox.showinfo("تم", "✅ تم دمج جميع الفيديوهات بنجاح!")
    except Exception as e:
        messagebox.showerror("خطأ", str(e))

# ===============================
# واجهة المستخدم
# ===============================

def run_gui():
    root = tk.Tk()
    root.title("🔊 دمج الفيديو مع الصوت المناسب (صوت مختلف لكل فيديو + Fade)")

    video_dir_var = tk.StringVar()
    audio_dir_var = tk.StringVar()
    output_dir_var = tk.StringVar()

    def browse_dir(var):
        var.set(filedialog.askdirectory())

    def update_progress(current, total):
        progress_var.set((current / max(1, total)) * 100)
        progress_label.config(text=f"{current} من {total} تمت معالجته")
        root.update_idletasks()

    def start():
        global cancel_processing
        cancel_processing = False
        if not video_dir_var.get() or not audio_dir_var.get():
            messagebox.showwarning("تحذير", "يرجى تحديد مجلدات الفيديوهات والأصوات.")
            return
        threading.Thread(target=process_all, args=(video_dir_var.get(), audio_dir_var.get(), output_dir_var.get() or "output", update_progress), daemon=True).start()

    def cancel():
        global cancel_processing
        cancel_processing = True
        messagebox.showinfo("إلغاء", "❌ جاري إيقاف العملية...")

    tk.Label(root, text="📁 مجلد الفيديوهات:").grid(row=0, column=0, sticky="e")
    tk.Entry(root, textvariable=video_dir_var, width=50).grid(row=0, column=1)
    tk.Button(root, text="اختيار", command=lambda: browse_dir(video_dir_var)).grid(row=0, column=2)

    tk.Label(root, text="🎵 مجلد الأصوات:").grid(row=1, column=0, sticky="e")
    tk.Entry(root, textvariable=audio_dir_var, width=50).grid(row=1, column=1)
    tk.Button(root, text="اختيار", command=lambda: browse_dir(audio_dir_var)).grid(row=1, column=2)

    tk.Label(root, text="💾 مجلد الحفظ:").grid(row=2, column=0, sticky="e")
    tk.Entry(root, textvariable=output_dir_var, width=50).grid(row=2, column=1)
    tk.Button(root, text="اختيار", command=lambda: browse_dir(output_dir_var)).grid(row=2, column=2)

    tk.Button(root, text="ابدأ المعالجة", bg="green", fg="white", width=20, command=start).grid(row=3, column=1, pady=10)
    tk.Button(root, text="إلغاء", bg="red", fg="white", width=20, command=cancel).grid(row=4, column=1, pady=10)

    progress_var = tk.DoubleVar()
    progress_label = tk.Label(root, text="")
    progress_label.grid(row=5, column=1)
    progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
    progress_bar.grid(row=6, column=0, columnspan=3, padx=20, pady=10, sticky="we")

    root.mainloop()

if __name__ == "__main__":
    run_gui()
