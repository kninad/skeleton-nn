# unet-skel-2d
Reproducing results for 2d Unet based skeleton extraction

## Setup

**Environment:**
The code assumes a python3 environment with `pytorch`. For full list of packages
see the `requirements.txt` (for pip) or `environment.yml` (for conda). The
preferred and tested method is via conda.

**Dataset:**
The preferred folder to place the datasets is under `./data/` (ignored by 
the `.gitignore` file) with symlink to the actual location on disk if needed.
The current training and evaluation scripts assume a binary image skeleton 
dataset under `./data/train/` (or elsewhere via the `specs.json` file):

```
./data/train/
    images/
        img_1.png
        img_2.png
    labels/
        img_1.png
        img_2.png
``` 

Note that this is just a sample for the binary image skeleton task. For
other uses, the dataset folder structure can change and hence make sure to
add a new `torch.utils.data.DataSet` and `torch.utils.data.DataLoader`
corresponding to it in `./utils/data.py`.

**Experiments:**
The training and evaluation scripts assume a certain format for running an
experiment with its associated hyperparameters. Each experiment should have an
*experiment directory* under `./experiments/` with a `specs.json` detailing the
network architecture, parameters, dataset path etc. Only the `specs.json` is
required before running an experiment.

Later, during training and evaluation, you will see the folders corresponding to
model parameters (`checkpoints`), optimizer parameters (`optim_parameters`)
-- each with data for certain epochs and a `*_latest.pth` file for the latest
epoch. Finally, there is also a folder for evaluation output (`evals`) where
output images are stored.

```
experiment/<exp_name>/
    specs.json
    checkpoints/
        model_checkpoint_<epoch>.pth
        model_checkpoint_latest.pth
    optim_parameters/
        optimizer_<epoch>.pth
        optimizer_latest.pth
    evals/
        images/
            test_out1.img
``` 

An example for a `specs.json` file. Note the `"DataSource"` key which should
point to the dataset location relative to the root of repo (or `trainer.py`). 

```json
{
  "Description" : [ "Sample ",
                    "description." ],
  "DataSource" : "./data/train/",
  "NetworkSpecs": {
    "channels": [1, 64, 128, 256, 512],
    "num_class": 1
  },
  "Epochs": 200,
  "SaveEvery": 40,
  "LearningRate": 0.005,
  "BatchSize": 8,
  "Debug": false,
  "NumDebug": 0 
}
```
Note the `"Debug"` and `"NumDebug"` keys which signal to the training script to
only use a small portion of the training data. This is useful to check for 
errors and overfitting the model on a small subset of the data.

## Training and Evaluation

See the `trainer.py` file for a sample training script and `eval.py` for a
sample evaluation script. With the dataset and experiment directory set up, run
the training script by providing the path to the experiment folder as a 
command line argument with `-e` or `--exp_dir` flags:

`python trainer.py -e experiments/<exp_dir>/`

The training script will write out `tensorboard` summary logs to the experiment
folder if you need to monitor the training along with saving the model and
optimizer parameters.

The `eval.py` script takes in an experiment folder (via `-e` or `--exp_dir` flags)
along with the epoch/checkpoint to evaluate via `-c` or `--checkpoint` flag. In
the current state, it loads up the saved model and outputs the predicted 
skeletons as images in `experiments/<exp_dir>/evals/images/`

`python eval.py -c latest -e experiments/<exp_dir>/`.
