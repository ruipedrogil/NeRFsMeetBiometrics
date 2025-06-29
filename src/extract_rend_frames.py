# Import required libraries for video processing and file operations
import cv2
import os
import argparse

# Main function to extract frames from video
def main():
    # Set up argument parser for command line interface
    parser = argparse.ArgumentParser(description="Extract frames uniformly from a video.")
    parser.add_argument('--video_path', type=str, required=True, help='Input video path')
    parser.add_argument('--output_path', type=str, required=True, help='Output directory for extracted frames')
    parser.add_argument('--num_frames', type=int, default=36, help='Number of frames to extract (default: 36)')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)  # Ensure output directory exists

    # Open video file
    cap = cv2.VideoCapture(args.video_path)  # Load video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    if total_frames == 0:
        print(f"[ERROR] Could not open video: {args.video_path}")
        return

    # Calculate frame extraction interval to get uniform distribution
    frame_step = max(1, total_frames // args.num_frames)

    # Initialize counters
    frame_count = 0
    saved_count = 0

    # Process video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Save frame if at the correct step and haven't saved enough yet
        if frame_count % frame_step == 0 and saved_count < args.num_frames:
            filename = os.path.join(args.output_path, f'frame_{saved_count:02d}.png')
            cv2.imwrite(filename, frame)
            saved_count += 1
        frame_count += 1

    # Release video resources
    cap.release()
    print(f'Total frames saved: {saved_count}')

if __name__ == "__main__":
    main()
