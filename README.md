# HIT Object Detection

## Install
Install the required libraries and packages using pip as shown below.
```
pip install -r requirements.txt
```
Next, install pytorch using the appropriate pip command from https://pytorch.org/get-started/locally/.
An example of this would be the command shown below.
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

## Run training for SSD on HIT dataset
Training can be started using the ```train.py``` script as shown below.
```
python train.py
```
There are a few parameters that can be changed at the top of the ```train.py``` script.