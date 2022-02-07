import cv2

from PIL import Image
from pathlib import Path


def save_gif(sfn, frames, fps):
    frames = [Image.fromarray(f) for f in frames]
    frame, *frames = frames
    ms_per_frame = int(1.0 / fps) * 1000
    frame.save(fp=sfn, format='GIF', append_images=frames,
               save_all=True, duration=ms_per_frame)


def save_video(sfn, frames, fps):
    sfn = Path(sfn)

    assert sfn.suffix in [".mp4", ".avi"], \
           f"Not supported video format: {sfn.extension}"

    if sfn.suffix == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # fourcc =  cv2.VideoWriter_fourcc(*"avc1")
    elif sfn.suffix == ".avi":
        fourcc =  cv2.VideoWriter_fourcc(*"XVID")

    height, width = frames[0].shape[:2]
    out = cv2.VideoWriter(str(sfn), fourcc, fps, (width, height), True)

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()

