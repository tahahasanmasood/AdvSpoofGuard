# AdvSpoofGuard: Adversarial Attack Robust Face Presentation Attack Detection System
This is the Pytorch implementation of the paper submitted/accepted/published in _______. The paper is available at this [link](https://)

### Abstract

### Proposed Unpaired Learning-based Adversarial Attack Generation Model

### Proposed CycleGAN-based Adversarial Attack Generation Model

### Architecture of the Proposed Face Presentation Attack Detection Classifier

### Results

## Usage

### Pre-Requisite

- Python: 3.10.4 or higher    
- Pytorch: 2.1.1 torchaudio==2.1.1 torchvision==0.16.1
- CPU or NVIDIA GPU

## Face PAD

### Dataset Placement
```bash
└── Your_Data_Dir
   ├── Replay_Attack 
   ├── Replay_Mobile
   ├── OULU_NPU
   ├── ROSE_Youtu
   └── ...
```
### Datasets Details

![datasets_details](https://github.com/user-attachments/assets/5f24bb33-5da1-46bd-b6b3-87abef4259d6)


### For Replay-Attack, Replay-Mobile, ROSE-Youtu, OULU-NPU Datasets

#### Before Adversarial Training
- Training: Run  `facePAD/before_adv_train/train.ipynb` 
- Testing: Run `facePAD/before_adv_train/test.ipynb`

#### After Adversarial Training
- Training: Run  `facePAD/after_adv_train/train_fake.ipynb` 
- Testing: Run `facePAD/after_adv_train/test_fake.ipynb`

### For OULU-NPU Dataset Testing
- Validation: To calculate the threshold and EER, run `facePAD/OULU_NPU/val.ipynb` 
- Testing: Run `facePAD/OULU_NPU/test.ipynb`

### Pre-trained Models
To apply pre-trained models, follow the folder `pre_trained_models`

### Videos to Images Conversion
Follow the folder `convert_images` and run `convert_images.ipynb`

### Plot t-Distributed Stochastic Neighbor Embedding (t-SNE)
To plot the t-SNE of all datasets, run `TSNE/datasets_TSNE.ipynb`

### Demo
- To demonstrate how to run the Face PAD codes for all datasets, we have provided an example demo code for the Replay_Attack dataset. 
- Training: To train the model from scratch, run `demo/train_Replay_Attack.ipynb`
- Testing: Load the pre-trained from `demo/RA_best_model.pth` and run `demo/test_Replay_Attack.ipynb`

## CycleGAN

### Dataset Preparation
```
.
├── datasets                   
|   ├── <dataset_name>         # i.e. Replay_Attack
|   |   ├── train              # Training
|   |   |   ├── A              # Contains domain A images (i.e. Real Attack)
|   |   |   └── B              # Contains domain B images (i.e. Real Bonafide)
|   |   └── test               # Testing
|   |   |   ├── A              # Contains domain A images (i.e. Real Attack)
|   |   |   └── B              # Contains domain B images (i.e. Real Bonafide)

```

### Training
```
python train.py --dataroot ./datasets/Replay_Attack --name replay_attack --model cycle_gan --display_id -1

```
- Once your model has trained, copy over the last checkpoint to a format that the testing model can automatically detect: Use `cp ./checkpoints/replay_attack/latest_net_G_A.pth ./checkpoints/replay_attack/latest_net_G.pth` if you want to transform images from class A to class B and `cp ./checkpoints/replay_attack/latest_net_G_B.pth ./checkpoints/replay_attack/latest_net_G.pth` if you want to transform images from class B to class A.
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097
- The visdom display functionality is turned on by default. To avoid the extra overhead of communicating with `visdom` set `--display_id -1`
- To see more intermediate results, check out ./checkpoints/replay_attack/web/images

### Testing
```
python test.py --dataroot datasets/Replay_Attack/testA --name replay_attack --model test --no_dropout

```

- Change the `--dataroot` and `--name` to be consistent with your trained model's configuration.
- The test results will be saved at `./results/replay_attack/latest_test/images`
- The option --model test is used for generating results of CycleGAN only for one side. This option will automatically set --dataset_mode single, which only loads the images from one set. On the contrary, using --model cycle_gan requires loading and generating results in both directions, which is sometimes unnecessary. The results will be saved at ./results/. Use --results_dir {directory_path_to_save_result} to specify the results directory.
- For your own experiments, you might want to specify --netG, --norm, --no_dropout to match the generator architecture of the trained model.

## Citation
If you use this code for your research, please cite our paper.
```
bibtex
```

## Acknowledgments
Our CycleGAN code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). All credit goes to the authors of [CycleGAN](https://arxiv.org/abs/1703.10593), Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros.



