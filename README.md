# M551M1 Elevator Counting with YOLOv8n

People-counting from top-down elevator views. Traditional weight sensors can’t distinguish between people and objects, leading to "ghost stops" (stopping when full) and safety risks. This project replaces guesswork with precise, real-time headcount data.

Using the Nuvoton M55M1 and its Ethos-U55 NPU, the system runs optimized ML models directly on the device. It converts a top-down camera feed into a digital count without the latency or cost of cloud processing.

100% data privacy by keeping all video processing local. Success is measured by achieving >95% accuracy and a smooth 15+ FPS, providing the speed and reliability needed for industrial building safety.

## Dataset

The dataset used is located at https://huggingface.co/datasets/bdanko/overhead-person-detection. To see how it was prepared, refer to https://github.com/bencejdanko/prepare-overhead-person-detection.

## Performance
  
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Precision (P)** | `93.2%` | Positive prediction accuracy (low false positives) |
| **Recall (R)** | `85.5%` | Sensitivity / detection coverage |
| **mAP50** | `93.5%` | Mean Average Precision at IoU=0.50 |
| **mAP50-95** | `54.8%` | Average mAP across thresholds |

### (NVIDIA Tesla T4 GPU)
| Operation | Latency (per image) | Equivalent Frame Rate |
| :--- | :--- | :--- |
| **Inference Only** | `1.3 ms` | **769 FPS** |
| **Full Pipeline** *(Pre + Inf + Post)* | `2.7 ms` | **370 FPS** |

## Developers

To train a new model refer to `Overhead_PersonDetection_Colab.ipynb` in the index directory.

Once you generate your `.tflite` model, replace it with the model at `m55m1-ElevatorCounting-YOLOv8n/Model/YOLOv8n-od.tflite`, and load to SRAM.



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

## Flashing the model 

### On WSL

```bash
# mount the drive
sudo mkdir -p /mnt/d
sudo mount -t drvfs D: /mnt/d
```

