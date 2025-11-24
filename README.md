**🌾 Cropify – Crop Recommendation System**


**📘 Overview**

Cropify is an end-to-end Machine Learning–powered crop recommendation system designed to assist users in selecting the most suitable crop based on soil and climate parameters. The system analyzes nutrient levels (N, P, K), temperature, humidity, rainfall, and soil pH to predict the optimal crop using trained ML models.

The project integrates multiple classification algorithms, provides a Streamlit-based interactive UI, and supports complete offline execution with the option to deploy the application online using Vercel.


**✨ Key Features**

1. Predicts the recommended crop based on real agricultural parameters
2. Utilizes multiple trained ML models (Random Forest, Decision Tree, Naive Bayes, MLP)
3. Fully interactive and responsive Streamlit web app
4. Real-time inference using .pkl model files
5. Clean modular architecture with separate model scripts
6. Easy deployment using Vercel + Streamlit wrapper
7. Works both offline (local PC) and online (cloud deployment)


**🧠 Machine Learning Models Used**

Cropify uses the following models trained on agricultural datasets:

Model                                       	Purpose                                	Notes
Random Forest Classifier	          Ensemble-based crop prediction	             High accuracy & stability
Decision Tree Classifier	          Simple interpretable model	                 Faster prediction, less generalization
Naive Bayes Classifier	              Probabilistic prediction	                     Works well with independent features
MLP (Multi-Layer Perceptron)	      Neural network for non-linear patterns 	     Best model for complex relationships

All trained models are stored as .pkl files and loaded in the Streamlit app at runtime.


**📂 Project Structure**

Cropify---ML/
│
├── cropii.py                      # Main Streamlit application
├── model_training.py              # Base training script
├── rfc2.py                        # Random Forest training
├── rtc2.py                        # Decision Tree training
├── mlpc2.py                       # MLP model training
├── nbc2.py                        # Naive Bayes training
├── MLP.pkl / random_forest.pkl    # Serialized model files
├── dataset.csv                    # Training dataset
├── crop_prediction_dataset.csv    # Additional dataset
├── assets/                        # Images and UI assets
└── translations/                  # Multilingual support


**🧰 Tech Stack**

Languages & Frameworks:
1. Python
2. Streamlit

Libraries:
1. NumPy
2. Pandas
3. Scikit-learn
4. Matplotlib
5. Seaborn
6. Joblib / Pickle
7. OpenCV
8. Pillow


**⚙️ Setup Instructions (Local Machine):**

1️⃣ Open project folder:
cd Cropify---ML

2️⃣ Create virtual environment:
python -m venv venv

3️⃣ Activate virtual environment:
Windows:
venv\Scripts\activate

4️⃣ Install dependencies:
pip install numpy pandas scikit-learn matplotlib seaborn requests joblib streamlit scikit-image opencv-python pillow

5️⃣ Run the ML model scripts (optional but recommended):
python rfc2.py
python rtc2.py
python nbc2.py
python mlpc2.py

6️⃣ Run the Streamlit app:
streamlit run cropii.py


**🚀 Deployment on Vercel (For Streamlit App)**

Vercel does not natively support Streamlit, but you can deploy it using a custom FastAPI wrapper or Docker.
Below is the simple, recommended method (FastAPI-based):

**🔧 Step 1: Install Deployment Dependencies**
In your project folder:

pip install fastapi uvicorn streamlit

**🗂️ Step 2: Create api/app.py**

Create folder api/ and inside it file app.py:

from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.get("/")
def run_app():
    subprocess.Popen(["streamlit", "run", "cropii.py", "--server.address", "0.0.0.0", "--server.port", "7860"])
    return {"status": "Streamlit app started"}

**📄 Step 3: Create vercel.json**

Add this file to your root directory:

{
  "version": 2,
  "builds": [
    { "src": "api/app.py", "use": "@vercel/python" }
  ],
  "routes": [
    { "src": "/(.*)", "dest": "api/app.py" }
  ]
}

**📤 Step 4: Push to GitHub & Deploy**

Upload your project to a GitHub repository
Visit Vercel dashboard
Click New Project → Import from GitHub
Select your repo
Click Deploy
Your app will be live on a Vercel domain like:
**https://cropify-yourname.vercel.app**


**📸 Screenshots (Add your own)**
assets/Cropify logo.png
assets/Wheat.jpg
assets/Rice.jpg
...

**👨‍💻 Author**

Abhishek Bajpai
Creator & Developer – Cropify

**📜 License**

This project is for educational and research purposes.
Free to use and modify, provided proper credit is given.



a project wiki,

or PowerPoint slides for your project viva.
