# Import required libraries for face detection and image processing
from retinaface import RetinaFace
from PIL import Image
import numpy as np
import os
from ultralytics import YOLO
import argparse

# Disable GPU usage by default
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Basic face cropping function using RetinaFace
# - Takes input and output paths
# - Returns True if successful, False if no face detected
def crop_face(input_path, output_path):
    im = Image.open(input_path).convert('RGB')  # Ensures RGB
    img_np = np.array(im)
    faces = RetinaFace.detect_faces(img_np)
    if not faces:
        print(f"No face detected in {input_path}.")
        return False
    # Get the largest detected face
    def area(face):
        x1, y1, x2, y2 = face['facial_area']
        return (x2 - x1) * (y2 - y1)
    largest_face = max(faces.values(), key=area)
    x1, y1, x2, y2 = largest_face['facial_area']
    cropped = im.crop((x1, y1, x2, y2))
    cropped.save(output_path)
    print(f"Cropped face saved at: {output_path}")
    return True

# Process all images in a folder with basic face cropping
def crop_faces_in_folder(folder_path):
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            in_path = os.path.join(folder_path, fname)
            out_path = os.path.join(folder_path, os.path.splitext(fname)[0] + "_face.png")
            try:
                im = Image.open(in_path).convert('RGB')  # Ensures RGB
                img_np = np.array(im)
                faces = RetinaFace.detect_faces(img_np)
                if not faces:
                    print(f"No face detected in {fname}.")
                    continue
                largest_face = max(faces.values(), key=lambda face: (face['facial_area'][2] - face['facial_area'][0]) * (face['facial_area'][3] - face['facial_area'][1]))
                x1, y1, x2, y2 = largest_face['facial_area']
                cropped = im.crop((x1, y1, x2, y2))
                cropped.save(out_path)
                print(f"Cropped face saved at: {out_path}")
            except Exception as e:
                print(f"Error processing {fname}: {e}")

# Advanced two-step cropping: first person detection with YOLO, then face detection
# - Uses YOLO for person detection
# - Then uses RetinaFace for face detection within person crop
# - Handles image resizing for small images
def crop_person_then_face(input_path, output_path, yolo_model_path='yolov8n.pt', conf_threshold=0.2):
    # 1. Detect person with YOLO
    yolo = YOLO(yolo_model_path)
    im = Image.open(input_path).convert('RGB')
    # Resize to 640x640 if smaller
    w, h = im.size
    resize_flag = False
    temp_input = None
    if w < 640 or h < 640:
        im_resized = im.resize((640, 640))
        temp_input = "_temp_input_yolo.png"
        im_resized.save(temp_input)
        yolo_input_path = temp_input
        resize_flag = True
    else:
        yolo_input_path = input_path
    results = yolo(yolo_input_path, conf=conf_threshold)
    if resize_flag and temp_input is not None and os.path.exists(temp_input):
        os.remove(temp_input)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    # Class 0 = person
    person_boxes = [box for box, cls in zip(boxes, classes) if int(cls) == 0]
    if not person_boxes:
        print(f"No person detected in {input_path}.")
        return False
    # Get the largest detected person
    box = max(person_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    # If resized, adjust coordinates
    if resize_flag:
        scale_x = w / 640
        scale_y = h / 640
        x1, y1, x2, y2 = [int(round(coord * scale)) for coord, scale in zip(box, [scale_x, scale_y, scale_x, scale_y])]
    else:
        x1, y1, x2, y2 = map(int, box)
    cropped_person = im.crop((x1, y1, x2, y2))
    # Save temporarily
    temp_path = "_temp_person_crop.png"
    cropped_person.save(temp_path)
    # 2. Detect and crop the face in the person crop
    im_person = Image.open(temp_path).convert('RGB')
    img_np = np.array(im_person)
    faces = RetinaFace.detect_faces(img_np)
    if not faces:
        print(f"No face detected after person crop in {input_path}.")
        os.remove(temp_path)
        return False
    def area(face):
        x1, y1, x2, y2 = face['facial_area']
        return (x2 - x1) * (y2 - y1)
    largest_face = max(faces.values(), key=area)
    fx1, fy1, fx2, fy2 = largest_face['facial_area']
    cropped_face = im_person.crop((fx1, fy1, fx2, fy2))
    cropped_face.save(output_path)
    os.remove(temp_path)
    print(f"Cropped face saved at: {output_path}")
    return True

# Process folder with two-step (person+face) cropping
def crop_person_then_face_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            in_path = os.path.join(input_folder, fname)
            name, ext = os.path.splitext(fname)
            out_path = os.path.join(output_folder, f"{name}_face{ext}")
            try:
                result = crop_person_then_face(in_path, out_path)
                if not result:
                    print(f"[WARNING] Could not crop face in: {fname}")
            except Exception as e:
                print(f"[ERROR] Failed to process {fname}: {e}")

# Main function with argument parsing
# - Supports three operation modes
# - Handles both single files and folders
# - Configurable detection threshold
def main():
    parser = argparse.ArgumentParser(description="Crop faces/people in images.")
    parser.add_argument('--mode', type=str, required=True, choices=['single', 'folder', 'person_face'], help='Operation mode: single (single image), folder (folder), person_face (person+face)')
    parser.add_argument('--input', type=str, required=True, help='Input image or folder path')
    parser.add_argument('--output', type=str, required=True, help='Output path for cropped images')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt', help='YOLO model path (optional)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for detection (default: 0.5)')
    args = parser.parse_args()

    if args.mode == 'single':
        # Do not create folder, save directly to the indicated path
        crop_face(args.input, args.output)
    elif args.mode == 'folder':
        os.makedirs(args.output, exist_ok=True)
        crop_faces_in_folder(args.input)
    elif args.mode == 'person_face':
        if os.path.isdir(args.input):
            os.makedirs(args.output, exist_ok=True)
            crop_person_then_face_in_folder(args.input, args.output)
        else:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            crop_person_then_face(args.input, args.output, args.yolo_model, args.threshold)
    else:
        print('Invalid mode.')

if __name__ == "__main__":
    main()

# Example usage for a single file:
# crop_face("/path/to/image.jpg", "/path/to/output_face.png")

# Example usage for a folder:
# crop_faces_in_folder("/path/to/frames/")

# Example usage for person+face:
# crop_person_then_face(
#      "/path/to/input.png",
#      "/path/to/output_face.png"
# )

# Example usage for a folder:
# crop_person_then_face_in_folder(
#      "/path/to/frames/",
#      "/path/to/face_frames/",
# )