import cv2 as cv
import os

video_folder = r"C:/Users/narin/Downloads/"  # folder containing multiple videos
output_root = "frames"         # master folder for all extracted frames
os.makedirs(output_root, exist_ok=True)

# Allowed video extensions
video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".MOV")

for file in os.listdir(video_folder):
    if file.lower().endswith(video_extensions):
        video_path = os.path.join(video_folder, file)
        video_name = os.path.splitext(file)[0]

        # Create a subfolder for this video's frames
        output_folder = os.path.join(output_root, video_name)
        os.makedirs(output_folder, exist_ok=True)

        print(f"\nProcessing {file} ...")

        cap = cv.VideoCapture(video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_name = f"frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_name)

            cv.imwrite(frame_path, frame, [int(cv.IMWRITE_JPEG_QUALITY), 100])
            print(f"Saved {frame_name}")
            frame_count += 1

        cap.release()

print("\nDone extracting frames from all videos!")
