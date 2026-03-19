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
# train
python dg_train.py --model-cfg relu6-yolov8.yaml --data /home/bence/datasets/top-down-people/data.yaml --imgsz 192 --weights yolov8n.pt --epochs 3 --device cpu

# test
# python dg_val.py --weights ./runs/train/exp2/weights/best.path --data /home/bence/datasets/top-down-people/data.yaml --img 192 --device cpu

# we train 192x192
# update exp{#} here
python nu_export_tflite_int8.py --format onnx --weights ./runs/train/exp5/weights/best.pt --img 192

# Generate calibration from dataset
python generate_calib_data.py \
  --img-size 192 192 \
  --n-img 4 \
  -o calib_data_192_n4_rgb.npy \
  --img-dir /home/bence/datasets/top-down-people/train

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

```

## datasets

https://app.roboflow.com/bences-workspace-zmeqo/top-down-people-mmue8-ogvbu

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