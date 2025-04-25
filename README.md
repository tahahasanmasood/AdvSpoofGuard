# AdvSpoofGuard: Optimal Transport Driven Robust Face Presentation Attack Detection System
This is the Pytorch implementation of the paper submitted in [**Knowledge-Based Systems**](https://www.sciencedirect.com/journal/knowledge-based-systems) journal. The paper is available at this [link](https://)

### Overview
AdvSpoofGuard presents a computationally efficient solution to address the vulnerability of face presentation attack detection (face PAD) against various attack types including physical, digital, and adversarial attacks by leveraging optimal transport (OT) and CycleGAN-based adversarial meta-learning.

In this repository, we present the code we use for the experiments in the paper. We provide the code to train the models, generate fake samples using OT-CycleGAN, and evaluate the quality of the fake samples with Fréchet Inception Distance (FID) and GMDM scores. The fake-generated samples by OT-CycleGAN are then utilized in adversarial training to enhance the robustness of face presentation attack detection (face PAD) systems. We also provide the test code and all pre-trained model weights to evaluate the proposed model's performance as reported in the paper.

### Proposed Unpaired Learning-based Adversarial Attack Generation Model
<p align="center">
  <img src="https://github.com/user-attachments/assets/e5e76b1a-ecfb-49b4-ab99-91c04d018e69" alt="Sample Image" width="650">
</p>

### Proposed OT-CycleGAN-based Adversarial Attack Generation Model
![cyclegan_architecture_final_revised_5](https://github.com/user-attachments/assets/dcf1583d-766d-4e51-87ae-30312ca9fe8b)

### Architecture of the Proposed Face Presentation Attack Detection Classifier
![FacePAD_architecture_final_revised_2](https://github.com/user-attachments/assets/0762afd9-744e-4ade-a21e-4f070da4e0d6)

### Results

![combined_table_new](https://github.com/user-attachments/assets/e9edbdea-8ee0-4dc8-bffc-ff9a29435524)

<p align="center">
  <img width="460" height="300" src="https://github.com/user-attachments/assets/363a7fd0-2397-48ae-b002-62ddffeaf274">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/765550d8-b674-40b4-be25-8ee734752065">
</p>

## Usage

### Pre-Requisite

- Python: 3.10.4 or higher    
- Pytorch: 2.1.1 torchaudio==2.1.1 torchvision==0.16.1
- CPU or NVIDIA GPU

*Note: the exact version of each package can be found in requirements.txt if necessary.*

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
All pre-trained models are available at this [link](https://drive.google.com/drive/folders/1BSeuv3eFFdJDVB9-imLAG0nlfpGORcoo?usp=sharing)

### Videos to Images Conversion
Follow the folder `convert_images` and run `convert_images.ipynb`

### Plot t-Distributed Stochastic Neighbor Embedding (t-SNE)
To plot the t-SNE for all datasets, run `facePAD/TSNE/datasets_TSNE.ipynb`

### Computational Complexity
- To compute the efficiency of different models, run `facePAD/computation_complexity/computational_efficiency/`
- To cacluate the HTER (%) on RY for different models, run `facePAD/computation_complexity/before_advtrain_RY and after_advtrain_RY`
- To plot the computational complexity graph for different models, run `facePAD/computation_complexity/computation_plot.ipynb`

### GradCAM Analysis
To visually analyze GradCAM for all datasets, run `facePAD/GradCAM/GradCAM.ipynb`

### Demo
- To demonstrate how to run the Face PAD codes for all datasets, we have provided an example demo code for the Replay_Attack dataset. 
- Training: To train the model from scratch, run `demo/train_Replay_Attack.ipynb`
- Testing: Load the pre-trained from `demo/RA_best_model.pth` and run `demo/test_Replay_Attack.ipynb`

## OT-CycleGAN
Our proposed OT-CycleGAN is heavily dependent on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We recommend that follow this repository for complete details of training and testing steps. For simplicity, we extracted some necessary steps for training and testing. 

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

### FID Score
To evaluate the quality of synthetic samples, we computed [FID score](https://github.com/mseitzer/pytorch-fid) to measure the similarity between original dataset samples and generated fake samples.

### GMDM Score
To evaluate the absolute value between two unpaired distributions, we computed [Generalized Multi-dimensional Distribution Overlap Metric (GMDM)](https://www.sciencedirect.com/science/article/abs/pii/S1051200423000258),
run `facePAD/GMDM/gmdm_evaluate.ipynb`


## Citation
If you use this code for your research, please cite our paper.
```
bibtex
```

## Acknowledgments
Our OT-CycleGAN code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Thanks to the authors of [CycleGAN](https://arxiv.org/abs/1703.10593) for sharing their code.



