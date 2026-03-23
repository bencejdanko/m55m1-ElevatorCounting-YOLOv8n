# M551M1 Elevator Counting with YOLOv8n

People-counting from top-down elevator views. Traditional weight sensors can’t distinguish between people and objects, leading to "ghost stops" (stopping when full) and safety risks. This project replaces guesswork with precise, real-time headcount data.

Using the Nuvoton M55M1 and its Ethos-U55 NPU, the system runs optimized ML models directly on the device. It converts a top-down camera feed into a digital count without the latency or cost of cloud processing.

100% data privacy by keeping all video processing local. Success is measured by achieving >95% accuracy and a smooth 15+ FPS, providing the speed and reliability needed for industrial building safety.

## Requirement

1. Keil uVision5

## Howto
1. Build by Keil
2. Copy Model/YOLOv8n-od.tflite file to SD card root directory.
3. Insert SD card to NUMAKER-M55M1 board
4. Run

## Performance
System clock: 220MHz
| Model |Input Dimension | ROM (KB) | RAM (KB) | Inference Rate (inf/sec) |  
|:------|:---------------|:--------|:--------|:-------------------------|
|YOLOv8n-od|192x192x3|2161|300| 34.0|

Total frame rate: 16 fps

## Developers

The model is located in `Model/YOLOv8n-od.tflite`.

To train a new model, use `yolov8_ultralytics`.

## Compiling a new model instructions

### Enviroment Requirements

#### For CPU

```bash
conda create --name yolov8_DG python=3.10 -y
conda activate yolov8_DG

python -m pip install --upgrade pip setuptools
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 ethos-u-vela --index-url https://download.pytorch.org/whl/cpu
```

#### For GPU

```bash
conda create --name yolov8_DG  python=3.10
conda activate yolov8_DG

python -m pip install --upgrade pip setuptools
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

python -m pip install .[export]
```

Then train:

```bash
# 1. Download and convert HuggingFace dataset to local YOLO format (90/10 Split)
# Saved to datasets/overhead_monochrome/
python download_hf_dataset.py --dataset bdanko/overhead-person-detection --out-dir datasets/overhead_monochrome

# 2. Train model using 192x192 resolution
python dg_train.py --model-cfg relu6-yolov8.yaml --data datasets/overhead_monochrome/data.yaml --imgsz 192 --weights yolov8n.pt --epochs 3 --device cpu

# 3. Export to tflite using onnx intermediary
# Update exp{#} to your highest run folder
python nu_export_tflite_int8.py --format onnx --weights ./runs/train/exp5/weights/best.pt --img 192

# 4. Generate calibration for tflite quantization from local monochrome dataset
python generate_calib_data.py \
  --img-size 192 192 \
  --n-img 4 \
  -o calib_data_192_n4_rgb.npy \
  --img-dir datasets/overhead_monochrome/train/images


# clear out old saved models
rm -rf saved_model

# generate new model
onnx2tf -i runs/train/exp5/weights/best.onnx \
  -oiqt \
  -cind images calib_data_192_n4_rgb.npy \
  "[[[[0,0,0]]]]" "[[[[1,1,1]]]]"

ls saved_model

cp saved_model/best_integer_quant.tflite vela/generated

cd vela/generated

vela best_integer_quant.tflite \
    --accelerator-config ethos-u55-128 \
    --output-dir . \
    --optimise Performance

# if you want custom cc   
# sudo apt install xxd
# xxd -i best_integer_quant_vela.tflite > best_integer_quant_vela.cc

# move replace the current KEIL model
cp /home/bence/m55m1-ElevatorCounting-YOLOv8n/yolov8_ultralytics/vela/generated/best_integer_quant_vela.tflite /home/bence/m55m1-ElevatorCounting-YOLOv8n/Model/YOLOv8n-od.tflite

## Datasets

The dataset is hosted on HuggingFace: [**`bdanko/overhead-person-detection`**](https://huggingface.co/datasets/bdanko/overhead-person-detection).

### Data Format & Preprocessing

The images in this dataset have already been processed to save processing time on-device:
- **Dimensions**: Constant **192x192** pixels (letterboxed).
- **Color space**: **Monochrome (Grayscale)** (1 channel).

To feed this into YOLO for training, use the downloaded script.
It loads dataset splits and transforms the absolute coordinate bboxes into normalized YOLO label structures:
  ```text
  <class_id> <x_center> <y_center> <width> <height>
  ```
- **Class ID**: `0` for `person`.






## Running inference

> [!WARNING]
> You must install additional libraries

Use `python install.py` to fetch the needed libraries, `Library` and `ThirdParty`.

install script from https://github.com/OpenNuvoton/ML_M55M1_SampleCode

Once you install, you'll get a `Library` and `ThirdParty` scripts.

You must configure their paths in `KEIL/ObjectDetection.csolution.yml`. By default they are:

```yaml
    - BSP_PATH: "C:/Library"
    - TP_PATH: "C:/ThirdParty"
```

## Shortcuts

```bash
rm -rf /mnt/c/Users/bence/m55m1-ElevatorCounting-YOLOv8n/

# copying wsl to windows
# to use vscode cmis plugin with keil
cp -r m55m1-ElevatorCounting-YOLOv8n/ /mnt/c/Users/bence
```