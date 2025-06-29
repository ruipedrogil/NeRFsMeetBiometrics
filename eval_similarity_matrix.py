import os
import subprocess
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Adjust defaults as needed for your structure
ROOT = os.path.join(os.path.dirname(__file__), 'renders')
PEOPLE = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
FRAMES_SUBDIR = 'face_frames'
TESTS_SUBDIR = 'tests_faces'

# Path to the verification script
FACE_REC_SCRIPT = os.path.join(os.path.dirname(__file__), 'face_recognition.py')

def main(args):
    global ROOT, FRAMES_SUBDIR, TESTS_SUBDIR, PEOPLE, FACE_REC_SCRIPT
    ROOT = args.root
    FRAMES_SUBDIR = args.frames_subdir
    TESTS_SUBDIR = args.tests_subdir
    OUTPUT_JSON = args.output_json
    N_TEST_IMAGES = args.n_test_images
    PEOPLE = [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
    FACE_REC_SCRIPT = os.path.join(os.path.dirname(__file__), 'face_recognition.py')

    results = {}

    for model_person in PEOPLE:
        frames_dir = os.path.join(ROOT, model_person, FRAMES_SUBDIR)
        if not os.path.isdir(frames_dir):
            continue
        results[model_person] = {}
        for test_person in PEOPLE:
            test_dir = os.path.join(ROOT, test_person, TESTS_SUBDIR)
            if not os.path.isdir(test_dir):
                continue
            # Define JSON path to pass and read
            json_path = OUTPUT_JSON
            cmd = [
                'python', FACE_REC_SCRIPT,
                '--frame_path', frames_dir,
                '--test_path', test_dir,
                '--output_json', json_path,
                '--threshold', '0.5'
            ]
            print(f"Evaluating: model={model_person} test={test_person}")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if os.path.exists(json_path):
                try:
                    if os.path.getsize(json_path) > 0:
                        with open(json_path) as f:
                            data = json.load(f)
                        results[model_person][test_person] = data.get('mean_similarity', None)
                    else:
                        print(f"[ERROR] Empty JSON: {json_path}")
                        results[model_person][test_person] = None
                except Exception as e:
                    print(f"[ERROR] Failed to read JSON {json_path}: {e}")
                    print(f"[STDERR] {proc.stderr}")
                    results[model_person][test_person] = None
                finally:
                    os.remove(json_path)
            else:
                print(f"[ERROR] JSON not found for {test_person} at {json_path}")
                print(f"[STDERR] {proc.stderr}")
                results[model_person][test_person] = None

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Similarity matrix saved in {OUTPUT_JSON}")

    # After saving JSON, generate plots
    # 1. Barplot per model
    for model_person in results:
        test_names = list(results[model_person].keys())
        sim_values = [results[model_person][t] if results[model_person][t] is not None else 0 for t in test_names]
        plt.figure(figsize=(10, 5))
        plt.bar(test_names, sim_values, color='skyblue')
        plt.ylim(0, 1)
        plt.ylabel('Mean Similarity')
        plt.title(f'Mean similarity - Model: {model_person}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{model_person}_barplot.png')
        plt.close()
    # 2. General heatmap
    all_models = list(results.keys())
    all_tests = list(next(iter(results.values())).keys()) if results else []
    matrix = np.array([[results[m][t] if results[m][t] is not None else 0 for t in all_tests] for m in all_models])
    plt.figure(figsize=(10, 8))
    im = plt.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(im, label='Mean Similarity')
    plt.xticks(ticks=np.arange(len(all_tests)), labels=all_tests, rotation=45)
    plt.yticks(ticks=np.arange(len(all_models)), labels=all_models)
    plt.title('Similarity Matrix (heatmap)')
    plt.tight_layout()
    plt.savefig('similarity_matrix_heatmap.png')
    plt.close()

    for model_person in results:
        print(f"\nModel: {model_person}")
        for test_person in results[model_person]:
            print(f"  Test: {test_person}  Mean: {results[model_person][test_person]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates a matrix of mean similarities between people in the dataset.")
    parser.add_argument('--root', type=str, default=os.path.join(os.path.dirname(__file__), 'renders'), help='Root directory with people folders')
    parser.add_argument('--frames_subdir', type=str, default='face_frames', help='Subfolder for NeRF frames')
    parser.add_argument('--tests_subdir', type=str, default='tests_faces', help='Subfolder for test images')
    parser.add_argument('--output_json', type=str, default='similarity_matrix.json', help='Output JSON file')
    parser.add_argument('--n_test_images', type=int, default=5, help='Number of test images to consider per person (default=5)')
    args = parser.parse_args()
    main(args)
