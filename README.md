# Point Transformers

## Dataset
   * https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip

## Weight Files
   * https://drive.google.com/drive/folders/1TUehCpCVpw8myE1Kj9Coe6z8QIPW0c9F?usp=sharing

## Training
1. In the ```config/config.yaml```, update the model name to be one of the four valid values.
    * Use ```Hengshuang``` for baseline.
    * Use ```Sumanu``` for Structured Pruning.
    * Use ```Luke``` for Dropout Extension.
    * Use ```Abhishek``` for Global Attention.

2. To use a checkpoint for training, download the weight from link under weight file section and copy the model in ```log\{name}\best_model.pth```.

3. Install requirements - pytorch, numpy, hydra, tqdm

4. Train ```python3 train.py```.

## Work Split
    * Luke wrote all lines of ```models/Luke/models.py``` and ```models/Luke/transformers.py```.
    * Sumanu wrote ```pruning_utils.py``` and ```train.py``` line 136-139. 
    * Abhishek wrote ```models/Abhishek/models.py``` lines 7-27, ```models/Abhishek/transformers.py``` lines 9-19, 49-61, 82-97,
      and added ```models/Abhishek/global_attention.py```.
    * For making baseline more fit for our use case:
      * Luke converted functions in ```provider.py``` from numpy to pytorch.
      * Abhishek updated training loop to reduce tensor movement between CPU/GPU.
      * Sumanu worked on cleaning the code and verifying the correctness of baseline.

## Powerpoint Presentation
   * https://drive.google.com/file/d/1lCmVzQ5uz5ZrKVPryV7g4u7N6_osuDrh/view?usp=sharing

## Video
   * https://drive.google.com/file/d/1LthlKgLSXVU3sNZkM1zkCltmAIVVmla_/view