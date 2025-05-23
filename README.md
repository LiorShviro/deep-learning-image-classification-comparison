# Deep Learning Image Classification Comparison

This project explores and compares multiple deep learning models for multi-class image classification using the [STL-10 dataset](https://cs.stanford.edu/~acoates/stl10/). It evaluates traditional and advanced architectures, focusing on accuracy, complexity, and the effectiveness of transfer learning.

## 📊 Dataset

- **[STL-10 dataset](https://cs.stanford.edu/~acoates/stl10/)**: A benchmark dataset containing 96×96 color images from 10 semantic classes (e.g., airplane, bird, car).
- For this project, images were resized to **64×64** to reduce computational load.
- Only the labeled training and test sets were used.

## 📈 Objective

To compare how various deep learning architectures and hyperparameter configurations affect image classification performance.

## 🧪 Models Implemented

1. **Logistic Regression**
   - Simple linear model with softmax activation.
   - Used as a baseline.

2. **Fully Connected Neural Network (FCNN)**
   - Three hidden layers with BatchNorm, Dropout, and ReLU.
   - ~6.4 million parameters.

3. **Convolutional Neural Network (CNN)**
   - Two convolutional layers with pooling and dropout, followed by fully connected layers.
   - ~1.1 million parameters.

4. **Pre-trained MobileNetV2 (Fixed)**
   - Used as a feature extractor with custom classification layers.

5. **Pre-trained MobileNetV2 (Fine-tuned)**
   - Full fine-tuning for better task-specific learning.

## 🛠️ Data Augmentation

Implemented using `torchvision.transforms`:
- `RandomHorizontalFlip` (p=0.5)
- `RandomCrop` to 64×64
- `ColorJitter` (p=0.3)

## 🔍 Hyperparameter Tuning

- **Learning Rates**: `0.01`, `0.001`
- **Optimizers**: `SGD`, `Adam`
- **Batch Sizes**: `32`, `64`
- **Epochs**: `5`, `10`, `15`, `20`

## 📊 Results

| Model                     | Best Test Accuracy |
|--------------------------|--------------------|
| Logistic Regression      | 17.04%             |
| Fully Connected NN       | 34.54%             |
| CNN                      | 41.76%             |
| Fixed MobileNetV2        | 73.20%             |
| Fine-tuned MobileNetV2   | 73.56%             |

## 📌 Conclusion

- **Fine-tuned MobileNetV2** yielded the best results, demonstrating the strength of transfer learning.
- **CNN** also performed well for a custom architecture.
- Simpler models showed limited performance, but offered useful baselines.

## 📁 Project Structure

All implementation was done in a **single Python notebook**. Supporting materials and visualizations are embedded within the notebook.

```
.
├── Deep_Learning_Image_Classification_Comparison.ipynb
├── Report.pdf
└── README.md
```

## 📝 Report

See the full PDF report in [Report.pdf](./Report.pdf) for model details, visualizations, and further analysis.

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/LiorShviro/deep-learning-image-classification-comparison.git
   cd deep-learning-image-classification-comparison
   ```

2. Open the notebook:
   ```bash
   jupyter notebook Deep_Learning_Image_Classification_Comparison.ipynb
   ```

3. Make sure you have the required packages installed:
   ```bash
   pip install torch torchvision matplotlib
   ```

---

## 🧠 Author

**Lior Shviro**  
GitHub: [@LiorShviro](https://github.com/LiorShviro)

---

## 📜 License

This project is licensed under the MIT License.
