# WIT-UAS: A Wildland-fire Infrared Thermal Dataset to Detect Crew Assets From Aerial Views

## About

![about](./figures/about.png)

- This dataset contains bounding box annotated Wildland-fire Infrared Thermal (WIT) images for crew assets detection with Unmanned Aerial Systems (UAS). It is captured during prescribed burns. The associated paper can be found [here](https://arxiv.org/pdf/2312.09159.pdf).

- [Available Labels](./dataset.classes)

## Test Object Detection

Clone the repo locally:

```Shell
git clone --recursive https://github.com/castacks/WIT-UAS-Dataset.git
```

### Environment Setup

There are 2 options:

- Docker (recommended):

  1. Install [**Docker**](https://docs.docker.com/get-docker/) and [**nvidia-docker2**](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

  1. Run the pre-built docker image (automatically pulls when missing):

     ```Shell
     ./scripts/run.sh
     ```

  1. Attach to the running container:

     ```Shell
     docker attach wit-uas-dataset
     ```

- Conda/Mamba environment manager:

  1. Install [**Anaconda**](https://docs.anaconda.com/anaconda/install/index.html) or [**Mamba**](https://mamba.readthedocs.io/en/latest/installation.html)

  1. Create environment for **WIT**:

     ```Shell
     mamba env create -f environment.yaml
     ```

  1. Activate **WIT**:

     ```Shell
     mamba activate wit-uas
     ```

### Visualization

- Visualization using wandb is optional, but when needed:

  1. Register an account on [wandb.ai](https://wandb.ai/site)

  1. Login to your wandb account locally:

     ```Shell
     wandb login
     ```

Send an email to [Mukai (Tom Notch) Yu](mailto:mukaiy@andrew.cmu.edu) if you are not invited to team *cmu-ri-wildfire*.

### Run training on WIT dataset

To train the models: `./model/yolo/train.py` or `./model/ssd/train.py`

You may need to adjust the batch size according to your GPU memory by using the `--batch-size` argument

## How To Contribute

1. Clone the repo locally

   ```Shell
   git clone --recursive https://github.com/castacks/WIT-UAS-Dataset.git
   ```

1. Create a new branch for your work:

   ```Shell
   git checkout -b <branch-name>
   ```

1. Setup required development environment

   ```Shell
   ./scripts/setup.sh
   ```

## Citation
If you find this repository useful for your work, please cite the following paper:

```bibtex
@INPROCEEDINGS{jong2023wit,
  author={Jong, Andrew and Yu, Mukai and Dhrafani, Devansh and Kailas, Siva and Moon, Brady and Sycara, Katia and Scherer, Sebastian},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={WIT-UAS: A Wildland-Fire Infrared Thermal Dataset to Detect Crew Assets from Aerial Views}, 
  year={2023},
  pages={11464-11471},
  url = {https://arxiv.org/pdf/2312.09159},
  doi={10.1109/IROS55552.2023.10341683}
}
```