# TNEA College Admission Predictor

## 📌 Project Overview
The **TNEA College Admission Predictor** is a **Streamlit-based web application** that helps students predict their chances of admission to various engineering colleges in Tamil Nadu based on **cutoff marks** and historical trends. It leverages **Machine Learning models** and **data visualization** to provide insights into college selections.

## 🚀 Features
- **User Input Form**: Allows users to enter their **cutoff marks** (Maths, Physics, Chemistry) and **community category**.
- **Cutoff Calculation**: Calculates the overall cutoff score based on input values.
- **College Prediction**: Uses historical cutoff data to suggest probable colleges.
- **Data Visualization**: Displays **cutoff trends** for different colleges and branches.
- **Interactive UI**: Built with **Streamlit** for an intuitive user experience.
- **Machine Learning Integration**: Implements **XGBoost Regression** to predict college admissions.

## 🛠️ Technologies Used
- **Python** 🐍
- **Streamlit** (Frontend UI)
- **Pandas** (Data Handling)
- **NumPy** (Numerical Computing)
- **XGBoost** (ML Model)
- **Scikit-Learn** (Evaluation Metrics)
- **Matplotlib & Seaborn** (Data Visualization)

## 📂 Project Structure
```
TNEA-Admission-Predictor/
│-- app.py                # Main Streamlit application
│-- data/
│   ├── cutoff_data.csv   # Historical cutoff data
│-- models/
│   ├── predictor.pkl     # Trained XGBoost model
│-- assets/
│   ├── images/           # UI assets & plots
│-- README.md             # Project Documentation
│-- requirements.txt      # Dependencies
```

## 🏗️ Installation & Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/TNEA-Admission-Predictor.git
   cd TNEA-Admission-Predictor
   ```
2. **Create a virtual environment (Optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For macOS/Linux
   venv\Scripts\activate     # For Windows
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## 🎯 Usage
- Open the **Streamlit UI** in the browser.
- Enter **Maths, Physics, and Chemistry** marks.
- Select **Community Category**.
- View **predicted colleges and cutoff trends**.

## 📊 Dataset
The dataset consists of **TNEA historical cutoff marks** for various engineering colleges and branches. It is used to train the prediction model and display past trends.

## 🤖 Machine Learning Model
- Uses **XGBoost Regression** to predict cutoffs and suggest colleges.
- Trained on historical **TNEA cutoff data**.
- **Mean Absolute Error (MAE)** is used as the evaluation metric.

## 📌 Future Enhancements
- Add **real-time updates** with new cutoff data.
- Improve prediction accuracy with **deep learning models**.
- Integrate **college placement insights**.

## 📝 License
This project is open-source under the **MIT License**.

---
✨ Developed by **[Your Name]** | Contact: your.email@example.com
