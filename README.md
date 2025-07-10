# Neural Radiance Fields Meet Biometrics Is it a Match?
Bachelor project based on the [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) framework using the Nerfacto method to create NeRF models for each person in the dataset. The project tests these models with images of different poses and performs face recognition using [FaceNet512](https://github.com/davidsandberg/facenet) to verify if the images correspond to the respective NeRF model.

---

You can download the full project report below:

[Download Report (PDF)](https://github.com/ruipedrogil/NeRFsMeetBiometrics/raw/main/report/report.pdf)

---

## Initial Requirements

This project is based on the Nerfstudio framework, so you need to follow its basic installation and setup.

---

## Installation with Pixi (Recommended)

Pixi is a fast and efficient environment manager built on top of the Conda stack. It is officially recommended by Nerfstudio.

### Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Clone and install Nerfstudio with Pixi

```bash
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
pixi run post-install
pixi shell
```

- This will install all dependencies including `colmap, tinycudann, and hloc`.
- Every time you want to use the project: `cd nerfstudio && pixi shell`

###  Run Nerfstudio example (optional)

```bash
pixi run train-example-nerf
```

### Use a specific version

```bash
git checkout tags/v1.1.3  # ou qualquer outra versão
```

---

## Method Code

Move the four Python script files in the source folder into the Nerfstudio folder to enable facial recognition.

#### Project folder structure:
The Python files should be placed inside the main nerfstudio folder (or in a subfolder like src/), resulting in the following structure:
```

nerfstudio/
├── ...
├── extract_rend_frames.py
├── crop_face.py
├── face_recognition.py
├── eval_similarity_matrix.py
├── src/                  # (Optional) For better script organization
│   ├── extract_rend_frames.py
│   ├── crop_face.py
│   ├── face_recognition.py
│   └── eval_similarity_matrix.py
├── ...

```

## Installing Additional Libraries

Inside the active pixi environment, install:

```bash
pip install deepface tf-keras ultralytics
pip install segment-anything torchvision opencv-python
pip install matplotlib fpdf2 scikit-learn
```

---

## Data Processing

### From a video:

```bash
export QT_QPA_PLATFORM=offscreen && ns-process-data video \
--data /home/socialab/nerfstudio/data/nerfstudio/subject/video/video.MOV \
--output-dir /home/socialab/nerfstudio/data/nerfstudio/subject
```

### From images:

```bash
export QT_QPA_PLATFORM=offscreen && ns-process-data images \
--data /home/socialab/nerfstudio/data/nerfstudio/subject/images \
--output-dir /home/socialab/nerfstudio/data/nerfstudio/subject
```

---

## Training with the nerfacto model

```bash
ns-train nerfacto --data /home/socialab/nerfstudio/data/nerfstudio/subject
```

---

## Video Rendering with Camera Path

```bash
ns-render camera-path \
--load-config outputs/subject/nerfacto/2025-06-26_121712/config.yml \
--camera-path-filename /home/socialab/nerfstudio/data/nerfstudio/subject/camera_paths/subject.json \
--output-path renders/subject/subject.mp4
```

> Note: This command is generated in Viser after selecting the keyframes.

---

## Viewing a Trained Model

```bash
ns-viewer --load-config /home/socialab/nerfstudio/outputs/subject/nerfacto/2025-06-25_180938/config.yml
```

---

## Evaluation of Results

```bash
ns-eval \
--load-config=/home/socialab/nerfstudio/outputs/subject/nerfacto/2025-06-26_143818/config.yml \
--output-path=/home/socialab/nerfstudio/outputs/subject/nerfacto/2025-06-26_143818/output.json
```

## Facial Recognition Scripts

### Extract frames from video:

```bash
python extract_rend_frames.py \
--video_path /home/socialab/nerfstudio/renders/subject/subject.mp4 \
--output_path /home/socialab/nerfstudio/renders/subject/frames
```

### Crop faces for folder:

```bash
python crop_face.py \
--mode person_face \
--input /home/socialab/nerfstudio/renders/subject/frames \
--output /home/socialab/nerfstudio/renders/subject/face_frames
```

### Crop test image:

```bash
python crop_face.py \
--mode single \
--input /home/socialab/nerfstudio/renders/subject/tests/subject1.png \
--output /home/socialab/nerfstudio/renders/subject/tests
```

### Run recognition test:

```bash
python face_recognition.py \
--frame_path /home/socialab/nerfstudio/renders/subject/face_frames \
--test_path /home/socialab/nerfstudio/renders/subject/tests_faces \
--output_json /home/socialab/nerfstudio/renders/subject/test_results/sim_cos.json
```

### Similarity matrix:

```bash
python eval_similarity_matrix.py \
--root /home/socialab/nerfstudio/renders \
--frames_subdir face_frames \
--tests_subdir tests_faces \
--output_json matriz_resultados.json
```

---

## Final Notes

- The pipeline is modular and can be adapted to different embedding models (e.g., ArcFace, DINOv2).
- The scripts should be placed inside the `nerfstudio` folder for easier execution.
- High-quality videos are recommended for best results.
- Can be used in applications such as avatar creation, biometric recognition, and re-identification.


