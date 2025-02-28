# Towards Efficient Large-Scale GNN Training Beyond the GPU Memory Limit via Micro-Batching

Our work is implemented based on DGL (with a PyTorch backend), and all experiments were conducted on Intel x86 processors and NVIDIA A100 GPUs.

## Install Requirements

Cherry is implemented using DGL 2.0.0 and PyTorch 2.0.1.

To be more specific, Cherry uses python 3.10, CUDA 11.8/cuDNN 8.7. You can quickly set up the environment needed for the experiments using several methods, and we recommend using Docker.

### Install with Docker

We have configured the necessary Docker images for the experiments, which can be obtained directly using the following command:

```shell
docker pull cherrywang/cherry:v1
```

### Install with Conda

We also prepared a quick configuration for the conda environment, which can be set up using the following command:

From requirements.sh

```shell
conda create -n env_name python=3.10
conda activate env_name
bash ./requirements.sh
```

From enviroment.yml

```shell
conda env create -f environment.yml
conda cherry
```

### Install Betty

If you want to test the effect of Betty at the same time, please execute the following command additionally

```shell
bash ./pytorch/mirco_batch_train/Betty_file/gen_data.sh
```

For more details about Betty's experimental environment, please refer to [Betty Code](https://zenodo.org/records/7439846)

## File Organization

- `pytorch/micro_batch_train` folder contains all the implementations of Cherry.

   - `cherry_graph_partitioner.py` realizes the Out-degree Centric Graph Partitioning

   - `c_block_dataloader.py` is the dataloader for Cherry-LMFG.

   - `micro_batch_train.py` is Cherry-GMFG training script which contains MBLs construction and training.

- `pytorch/models` folder contains all models used in our experiments.

- `pytorch/utils` folder contains some of our custom profiling tools and dataset loading.

- `pytorch/micro_batch_train/Betty_file` is Betty code from [Betty Code](https://zenodo.org/records/7439846), more details please read `pytorch/micro_batch_train/Betty_file/readme.md`

- `Evaluation` contains some of our experimental scripts and running examples of Cherry-LMFG and Cherry-GMFG.

## Run Cherry on your device

1. `git clone https://github.com/anonymousFor-asp/Cherry.git`
2. Configure your environment according to the install requirements
3. Configure your dataset storage folder in `pytorch/micro_batch_train/utils/load_graph.py`
   - All datasets used in the experiment can be automatically downloaded except Amazon.You can manually download the amazon dataset from [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT)

4. `./Evaluation` contains some test examples, including the writing of test scripts. You can adjust the parameters to test your own experiments.

All experimental scripts are easily extensible, and you can freely choose the datasets and model parameters you want to test.

### Run Cherry-LMFG

```shell
cd Evaluation/
bash run_LMFG.sh
```

### Run Cherry-GMFG

```shell
cd Evaluation/
bash run_GMFG.sh
```