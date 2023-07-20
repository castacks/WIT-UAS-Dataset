# HIT Object Detection

## Configure Environment

1. Install **Anaconda** following the [**official guide**](https://docs.anaconda.com/anaconda/install/index.html).

1. Create environment for **HIT**:

   ```shell
   conda env create -f environment.yml
   ```

1. Activate **HIT**:

   ```shell
   conda activate HIT
   ```

1. Login to your [wandb account](https://wandb.ai/site) locally:

   ```shell
   wandb login
   ```

Send an email to [Mukai (Tom Notch) Yu](mailto:mukaiy@andrew.cmu.edu) if you are not invited to team *cmu-ri-wildfire*.

## Run training on HIT dataset

Training can be started using the `yolo_train.py` or `ssd_train.py`:

```shell
python yolo_train.py
```

or

```shell
python ssd_train.py
```

You may need to adjust the batch size according to your GPU memory.
