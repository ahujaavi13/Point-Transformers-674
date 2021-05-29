# Point Transformers

## Dataset
   * https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip

## Weight Files
   * https://drive.google.com/drive/folders/1f1UwhSvjz6K8hzI9MgCfxAR-63gcjFO7?usp=sharing

## Training
1. In the ```config/config.yaml```, update the model name to be one of the four valid values.
    * Use ```Hengshuang``` for baseline.
    * Use ```Sumanu``` for Structured Pruning.
    * Use ```Luke``` for Dropout Extension.
    * Use ```Abhishek``` for Global Attention.

2. To use a checkpoint for training, download the weight from link under weight file section and copy the model in ```log\{name}\best_model.pth```.

3. Install requirements - pytorch, numpy, hydra, tqdm

4. Train ```python3 train.py```.

## Project Report
   * https://drive.google.com/file/d/1EgnxtvYnqaTuwcpw6_jecLRw8OATOokD/view?usp=sharing

## Powerpoint Presentation
   * https://drive.google.com/file/d/1lCmVzQ5uz5ZrKVPryV7g4u7N6_osuDrh/view?usp=sharing

## Video
   * https://drive.google.com/file/d/1LthlKgLSXVU3sNZkM1zkCltmAIVVmla_/view
