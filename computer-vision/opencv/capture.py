import cv2 as cv
import os

video_path=r"C:/Users/narin/Downloads/Video003.MP4"
output_folder="frames"
os.makedirs(output_folder, exist_ok=True)

cap=cv.VideoCapture(video_path)
frame_count=0
while(True):
    ret,frames=cap.read()
    if not ret:
        break

    frame_name=f"frame_{frame_count:04d}.jpg"
    frame_path=os.path.join(output_folder,frame_name)

    cv.imwrite(frame_path,frames,[int(cv.IMWRITE_JPEG_QUALITY),100])

    print(f"saved {frame_name}")
    frame_count+=1

cap.release()