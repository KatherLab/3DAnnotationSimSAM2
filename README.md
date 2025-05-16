# Overview
This is the code that was used to create the SAM 2-assisted segmentation masks in paper "Reducing Manual Workload in CT and MRI Annotation with the Segment Anything Model 2". This code only works in combination with the original SAM 2 repository (see below).
# Installation
```
git clone https://github.com/facebookresearch/sam2.git
cd sam2
git checkout 7e1596c0b6462eb1d1ba7e1492430fed95023598
```
then, move the folder "annotation_simulation" of this repo into the root folder of the sam2 repo
```
project_root
└── annotation_simulation
└── assets
...
```

Create a virutal enrironment (tested for Python 3.11.2) and install the dependencies from the original SAM 2 repository and additional packages
```
pip install -e .
pip install "pandas>=2.2.2" "monai>=1.3.2" "SimpleITK>=2.3.1" "matplotlib>=3.9.1" "scipy>=1.14.0" "torch==2.4.0" "torchvision==0.19.0"
python setup.py build_ext --inplace # see https://github.com/facebookresearch/sam2/issues/25
```
# Download SAM 2 weights
```
cd project_root/checkpoints
bash download_ckpts.sh 
```
# Data preparation
To simulate SAM 2-assisted annotation, you need a dataset organized in the following structure:
```
dataset_root/
└── SAM_2_frames_4/
    ├── ImagesTr/
    │   └── patient_x/
    │       ├── 0.png # single CT/MRI slice, as 8bit png
    │       ├── 1.png
    │       ├── ...
    │       └── coordinates_center.csv
    │   └── patient_x+1/
    │       ├── 0.png
    │       ├── 1.png
    │       ├── ...
    │       └── coordinates_center.csv
    |   ...
    ├── labeled_labelsTr/
    │   ├── patient_x_cls_1.nii.gz
    │   └── patient_+1_cls_1.nii.gz
    |   ...
    └── labelsTr_cls_1/
        ├── patient_x.nii.gz
        └── patient_x+1.nii.gz
        ...
```
#### ImagesTr/

Contains 8-bit PNG slices from CT or MRI scans.

Each subfolder (e.g., patient_x/) represents a single 3D scan, with each image (0.png, 1.png, ...) corresponding to a slice.

coordinates_center.csv

Located inside each patient folder and specifies the center slice and initial prompt coordinates used by SAM 2. The box coordinates were derived by the ground truth segmentation masks.

example:
| class | label_id | bbox_x1 | bbox_y1 | bbox_x2 | bbox_y2 | Description                     |
|-------|----------|---------|---------|---------|---------|---------------------------------|
| 1     | 1        | 81      | 229     | 89      | 238     | Box prompt for first tumor      |
| 1     | 2        | 177     | 309     | 197     | 336     | Box prompt for second tumor     |

...

#### labeled_labelsTr/

Contains labeled 3D segmentation masks (.nii.gz) for one specific class (e.g. liver tumors).

Format: patient_x_cls_1.nii.gz

Values:

    0 = background

    1, 2, 3, ... = connected tumor components (each component has a unique integer label)

#### labelsTr_cls_1/

Contains binary segmentation masks for one class.

Format: patient_x.nii.gz

Values:

    0 = background

    1 = foreground (e.g. tumor tissue)

# Adjust code
if you have extracted the slices as png, you have to adjust project_root/sam2/utils/misc.py
change line
```
if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
```
to
```
if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", '.png']
```
# Run annotation simulation
For your preprocessed dataset and dice_thresh (tau parameter in the paper), run
```
python project_root/annotation_simulation/video_prediction.py --prompt box --dice_thresh 0.9 --dataset_path /path/to/dataset_root
```
The script outputs the segmentation masks with which you can then train a segmentation model, e.g. using nnUNet (see their repository for more details).

# Reference
If you find our work useful in your research or if you use parts of this code please consider citing our publication.
TODO