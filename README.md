# DensResNet: Ethiopian Cuisine Classifier 🍽️🇪🇹

**DensResNet** is a deep learning-powered web application for classifying 11 traditional Ethiopian dishes using an ensemble of two powerful CNN architectures: **DenseNet121** and **ResNet50V2**. The model uses **soft voting** to combine predictions and improve classification accuracy.

---

## 🚀 Demo

Upload an image of Ethiopian food and the model will return top-2 predictions from:
- **DenseNet121**
- **ResNet50V2**
- **Soft Voting Ensemble**

It also shows bar chart visualizations of all class probabilities and a performance summary table.

---

## 📷 App Interface

| Upload Section | Prediction Output | Probability Charts |
|----------------|-------------------|--------------------|
| ![upload](images/interface1.png) | ![results](images/interface2.png) | ![charts](images/interface3.png) |

---

## 🧠 Models Used

- **DenseNet121** trained on Ethiopian food dataset  
- **ResNet50V2** trained on the same dataset  
- **Soft Voting** ensemble strategy for improved prediction confidence

---

## 📊 Test Accuracy

| Model        | Test Accuracy |
|--------------|---------------|
| DenseNet121  | 83.12%        |
| ResNet50V2   | 86.36%        |
| Ensemble     | **88.31%**    |

---

## 🗂️ Classes

The model classifies the input into one of the following 11 Ethiopian dishes:

beyaynetu, chechebsa, doro_wat, firfir, genfo,kiki1, kitfo, shekla_tibs, shiro_wat, tihlo, tire_siga

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Ethiopian_cuisine_classification.git
   cd Ethiopian_cuisine_classification
   ```
2. **Install dependencies:**:   
   ```commandline
    pip install -r requirements.txt
   ```
   ```
3. **Run the Streamlit app:**:   
   ```commandline
    streamlit run denserestnet_app.py
   ```

## 📐 Input Format
Image Size: 512×512 (automatically resized)

Accepted Formats: .jpg, .jpeg, .png

## 👨‍💻 Author

Yoseph Negash

📧 yosephn22@gmail.com

📅 2025
