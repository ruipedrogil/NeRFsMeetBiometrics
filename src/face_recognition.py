# Import necessary libraries for face recognition, image processing, and report generation
from deepface import DeepFace
from numpy import dot
from numpy.linalg import norm
import numpy as np
import os
from fpdf import FPDF
import cv2
import argparse
import json
import matplotlib.pyplot as plt

# Function to preprocess test images to match reference frames
# - Resizes to reference dimensions
# - Adjusts contrast and brightness
# - Applies Gaussian blur for noise reduction
def preprocess_test_face(img, ref_shape):
    # Resize to the frame's shape
    img = cv2.resize(img, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_AREA)
    # Adjust contrast and brightness
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    # Apply light blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


# Function to generate PDF report with comparison results
# - Creates structured report with test images, similarity scores, and match status
# - Includes reference frame examples
# - Supports custom output path
def gerar_pdf_resultado(frame_files, results, COS_SIM_THRESHOLD, output_path="face_comparison_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Face Similarity Report", 0, 1, 'C')
    pdf.ln(5)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Threshold: {COS_SIM_THRESHOLD}", 0, 1)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 10, "Test Image", 1)
    pdf.cell(40, 10, "Similarity", 1)
    pdf.cell(30, 10, "Result", 1, 1)
    pdf.set_font("Arial", '', 12)
    for img_path, cos_sim in results:
        pdf.cell(60, 10, os.path.basename(img_path), 1)
        pdf.cell(40, 10, f"{cos_sim:.4f}", 1)
        pdf.cell(30, 10, "MATCH" if cos_sim > COS_SIM_THRESHOLD else "NO MATCH", 1, 1)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Reference Frames (up to 3 examples):", 0, 1)
    for i, frame in enumerate(frame_files[:3]):
        pdf.image(frame, x=10 + i*65, y=pdf.get_y(), w=60)
    pdf.output(output_path)
    print(f"PDF generated at: {output_path}")

# Core function to compare reference frames with a test image
# - Calculates mean embedding from reference frames
# - Computes cosine similarity with test image embedding
# - Returns match status based on threshold
def compare_frames_with_test_image(frame_files, test_image_path, threshold=0.5):
    """
    Calculates cosine similarity between the mean embedding of frames and a test image.
    Prints the value and whether it is a MATCH or NO MATCH.
    """
    # Calculate embeddings for frames
    embeddings = []
    for frame_path in frame_files:
        try:
            emb = DeepFace.represent(img_path=frame_path, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
            embeddings.append(emb)
        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
    if len(embeddings) == 0:
        print("No embedding extracted from frames!")
        return None
    mean_embedding = np.mean(embeddings, axis=0)
    # Preprocess test image
    img_teste = cv2.imread(test_image_path)
    ref_img = cv2.imread(frame_files[0])
    if img_teste is not None and ref_img is not None:
        img_teste_proc = preprocess_test_face(img_teste, ref_img.shape)
        temp_path = test_image_path + "_proc.png"
        cv2.imwrite(temp_path, img_teste_proc)
        embedding_teste = DeepFace.represent(img_path=temp_path, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
        os.remove(temp_path)
        cos_sim = dot(mean_embedding, embedding_teste) / (norm(mean_embedding) * norm(embedding_teste))
        print(f"Cosine similarity: {cos_sim:.4f}")
        if cos_sim > threshold:
            print("MATCH")
            return cos_sim, "MATCH"
        else:
            print("NO MATCH")
            return cos_sim, "NO MATCH"
    else:
        print(f"Error reading {test_image_path} or reference {frame_files[0]}")
        return None

# Main execution block with argument parsing
# - Supports multiple comparison modes
# - Handles PDF and JSON output generation
# - Includes visualization capabilities
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face similarity verification with NeRF frames.")
    parser.add_argument('--frame_path', type=str, help='Folder with reference frames (NeRF)')
    parser.add_argument('--test_path', type=str, help='Folder with test images')
    parser.add_argument('--single_test_image', type=str, help='Single test image for direct comparison')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for MATCH/NO MATCH (default=0.5)')
    parser.add_argument('--output_pdf', nargs='?', const=True, default=False, help='If present, generates PDF with results. Can indicate path or just the flag.')
    parser.add_argument('--output_json', nargs='?', const=True, default=False, help='Path to JSON file to save results. If only the flag, uses default name.')
    args = parser.parse_args()

    if args.frame_path and args.test_path:
        # Batch processing mode for multiple test images
        frames_dir = args.frame_path
        test_dir = args.test_path
        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('_face.png')])
        test_files = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('_face.png')])
        if len(frame_files) == 0:
            print("No reference frame found!")
            exit()
        if len(test_files) == 0:
            print("No test image found!")
            exit()
        embeddings = []
        for frame_path in frame_files:
            try:
                emb = DeepFace.represent(img_path=frame_path, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
                embeddings.append(emb)
                print(f"Embedding extracted from: {frame_path}")
            except Exception as e:
                print(f"Error processing {frame_path}: {e}")
        if len(embeddings) == 0:
            print("No embedding extracted from frames!")
            exit()
        mean_embedding = np.mean(embeddings, axis=0)
        results = []
        ref_img_shape = None
        if frame_files:
            ref_img = cv2.imread(frame_files[0])
            if ref_img is not None:
                ref_img_shape = ref_img.shape
        for img_teste_path in test_files[:5]:
            img_teste = cv2.imread(img_teste_path)
            if ref_img_shape is not None and img_teste is not None:
                img_teste_proc = preprocess_test_face(img_teste, ref_img_shape)
                temp_path = img_teste_path + "_proc.png"
                cv2.imwrite(temp_path, img_teste_proc)
                embedding_teste = DeepFace.represent(img_path=temp_path, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
                os.remove(temp_path)
                cos_sim = dot(mean_embedding, embedding_teste) / (norm(mean_embedding) * norm(embedding_teste))
                print(f"Cosine similarity ({os.path.basename(img_teste_path)}): {cos_sim:.4f}")
                results.append({"image": img_teste_path, "similarity": float(cos_sim), "match": bool(cos_sim > args.threshold)})
            else:
                print(f"Error reading {img_teste_path} or reference {frame_files[0]}")
        # Mean similarity calculation and visualization
        if results:
            mean_sim = np.mean([r["similarity"] for r in results])
            print(f"Mean cosine similarity: {mean_sim:.4f}")
            # Custom bar chart: each tested image, green bar if >= threshold, red if < threshold, horizontal line for threshold
            sim_vals = [r["similarity"] for r in results]
            img_labels = [os.path.basename(r["image"]) for r in results]
            colors = ['green' if s >= args.threshold else 'red' for s in sim_vals]
            plt.figure(figsize=(max(7, len(sim_vals)), 4))
            bars = plt.bar(img_labels, sim_vals, color=colors)
            plt.axhline(args.threshold, color='blue', linestyle='--', label=f'Threshold ({args.threshold})')
            plt.ylim(0, 1)
            plt.ylabel('Cosine Similarity')
            plt.xlabel('Test Image')
            plt.title('Similarity by Test Image')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig('similarity_barplot.png')
            plt.close()
        else:
            mean_sim = None
        # PDF generation
        if args.output_pdf:
            if isinstance(args.output_pdf, str):
                pdf_path = args.output_pdf
            else:
                pdf_path = os.path.join(test_dir, "face_comparison_report.pdf")
            pdf_results = [(r["image"], r["similarity"]) for r in results]
            gerar_pdf_resultado(frame_files, pdf_results, args.threshold, pdf_path)
        # JSON output
        if args.output_json:
            if isinstance(args.output_json, str):
                json_path = args.output_json
            else:
                json_path = "face_comparison_results.json"
            output_dir = os.path.dirname(json_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump({"results": results, "mean_similarity": mean_sim}, f, indent=2)
            print(f"Results saved in {json_path}")
    elif args.frame_path and args.single_test_image:
        # Single image comparison mode
        frame_files = sorted([os.path.join(args.frame_path, f) for f in os.listdir(args.frame_path) if f.endswith('_face.png')])
        if len(frame_files) == 0:
            print("No reference frame found!")
            exit()
        cos_sim_result = compare_frames_with_test_image(frame_files, args.single_test_image, threshold=args.threshold)
        # Bar chart for single image
        if cos_sim_result is not None:
            sim_val = cos_sim_result[0]
            color = 'green' if sim_val >= args.threshold else 'red'
            plt.figure(figsize=(3,4))
            plt.bar([os.path.basename(args.single_test_image)], [sim_val], color=color)
            plt.axhline(args.threshold, color='blue', linestyle='--', label=f'Threshold ({args.threshold})')
            plt.ylim(0, 1)
            plt.ylabel('Cosine Similarity')
            plt.xlabel('Test Image')
            plt.title('Similarity (Single Image)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('similarity_barplot.png')
            plt.close()
        # PDF generation for single image
        if args.output_pdf and cos_sim_result is not None:
            if isinstance(args.output_pdf, str):
                pdf_path = args.output_pdf
                img_path = args.single_test_image
            else:
                img_path = args.single_test_image
                pdf_path = os.path.splitext(img_path)[0] + "_comparison_report.pdf"
            fake_results = [(img_path, float(cos_sim_result[0]))]
            gerar_pdf_resultado(frame_files, fake_results, args.threshold, pdf_path)
        # JSON output for single image
        if args.output_json and cos_sim_result is not None:
            img_path = args.single_test_image  # Ensure img_path is always defined
            if isinstance(args.output_json, str):
                json_path = args.output_json
            else:
                json_path = "face_comparison_results.json"
            output_dir = os.path.dirname(json_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump({"image": img_path, "similarity": float(cos_sim_result[0]), "match": cos_sim_result[1]}, f, indent=2)
            print(f"Result saved in {json_path}")
    elif args.output_json:
        # Empty JSON creation if only output flag is provided
        if isinstance(args.output_json, str):
            json_path = args.output_json
        else:
            json_path = "face_comparison_results.json"
        output_dir = os.path.dirname(json_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump({}, f, indent=2)
        print(f"Empty JSON file created at {json_path}")
    else:
        print("Use --frame_path and --test_path for multiple images, or --frame_path and --single_test_image for direct comparison. Use --output_pdf and/or --output_json to save results.")