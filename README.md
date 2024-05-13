
## DiffRA: General-Purpose Restorative Adversarial Attack based on Diffusion Model </sub>



## Dependenices

* OS: Ubuntu 20.04
* nvidia :
	- cuda: 11.7
	- cudnn: 8.5.0
* python3
* pytorch >= 1.13.0
* Python packages: `pip install -r requirements.txt`

## How to use our Code?

Here we provide an example for **image deraining** task, but can be changed to any problem with replacing the dataset.

We retrained the deraining model from scratch on a single RTX 4090 GPU.

The pretrained models for all tasks are provided [here](https://drive.google.com/drive/folders/1IsZsY4m8SYYvSSGJQoyky5Uajp9YigVs)

### Dataset Preparation

We employ Rain100H datasets for training (totally 1,800 images) and testing (100 images). 

Process datasets in a way such that rain images and no-rain images are in separately directories, as

```bash
#### for training dataset ####
datasets/rain/trainH/GT
datasets/rain/trainH/LQ


#### for testing dataset ####
datasets/rain/testH/GT
datasets/rain/testH/LQ

```

Then get into the `codes/config/deraining-attack` directory and modify the dataset paths in option files in 
`options/train/r-attack.yml` and `options/test/r-attack.yml`.


### Train
The main code for training is in `codes/config/deraining-attack` and the core algorithms for IR-SDE is in `codes/utils/sde_utils.py`.

You can train the model following below bash scripts:

```bash
cd codes/config/deraining-attack

# For single GPU:
python3 train.py -opt=options/train/r-attack.yml

```


### Evaluation
To evaluate our method, please modify the benchmark path and model path and run

```bash
cd codes/config/deraining-attack
python test.py -opt=options/test/r-attack.yml
```




## Citations
If our code helps your research or work, please consider citing our paper.
The following are BibTeX references:

```

```

---



#### --- Thanks for your interest! --- ####
