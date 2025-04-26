### W celu poprawnego działania, należy pobrać do folderu root dane z [kaggle.com](https://www.kaggle.com/competitions/facial-keypoints-detection/data)

# Project - Projektowanie i zastosowania sieci neuronowych
This project was created as part of university coursework for the subject "Design and Applications of Neural Networks". It allows to train, load and visualize implemented neural network model for the task of Facial Keypoint Recognition.

# Python Packages
- **PyTorch** (torch, torchvision) - main NN library
- **NumPy**
- **Pandas** - loading data
- **scikit-learn** - splitting the data to train and validation datasets
- **opencv-python** - camera visualization
- **matplotlib** - other visualizations
- **tensorboard** - logging training progress

# Installation
1. Clone the repository:
```bash
git clone https://github.com/mikolaj-paz/Projekt-PSN-Czesc-2.git
```
2. Navigate to the project directory:
```bash
cd Projekt-PSN-Czesc-2
```
3. Install required libraries:
```bash
pip install torch torchvision numpy pandas scikit-learn opencv-python matplotlib tensorboard
```

# Standard Usage
Run the main program:
```bash
python main.py
```
Here you have a few options to choose from. If you want to start training, make sure to read the next chapter.
```plaintext
1. Training
2. Downloading the model from file and showing sample.
3. Visualization through webcam.
4. Exit
```
>[!WARNING]
>Running the training loop overwrites ```model_weights.pth``` file, which is also used in visualizations (make sure to create a backup).

# Model Training
Running training requires a few preparation steps.

1. Downloading the training data from [kaggle.com](https://www.kaggle.com/competitions/facial-keypoints-detection/data) and putting a ```training.csv``` file into the directory ```dane/``` -> finally you have ```dane/training.csv``` in the root folder.
2. Running ```python splittraindata.py``` in terminal.
Your final file structure should look like this:
```plaintext
Projekt-PSN-Czesc-2/
├── dane/
|   ├── training.csv
|   ├── train_split.csv
|   └── val_split.csv
├── model_weights.pth
├── main.py
└── ...
```
3. Run ```main.py``` and choose the training option.
