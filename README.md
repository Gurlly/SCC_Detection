# 💻🔧Project Setup

### 📱 **Project Overview:**
This thesis project, developed by Computer Science students from the University of Santo Tomas, introduces a beta-stage system designed to streamline defense demonstrations through a user-friendly interface. Users can upload an image and choose between classification modes—with or without normalization. The system then generates Grad-CAM visualizations for each model, along with their respective classification results and confidence scores.

----

### 🩴 **Steps:**
1. Open separate terminals for /backend and /frontend
2. Download the models in [Models Link](https://drive.google.com/drive/folders/1TDydkN-F6dV9K6QxT1H3vt0Sn69nZ_ih?usp=drive_link)
3. Inside the `/backend` folder create a `/models` folder. Then place the models to its corresponding folders (*e.g., backend/models/coatnet/coatnet_normalized.pth*).
4. **For Backend:**
``` bash
python -m venv venv         # To create a virtual environment (optional)
venv\Scripts\Activate.ps1   # To activate the virtual environment (optional)
pip install -r requirements.txt
python run.py               # To run the backend server
```
5. **For Frontend:**
``` bash
npm install
npm run dev     # To launch web application
```
6. Open your browser and enter `localhost:5173`.
7. Upload an image (Margin Negative or Margin Positive).
8. Select classification type *w/ Normalization* or *w/o Normalization*.

----

### 👨‍👩‍👧‍👦 **Authors:**
- David Raniel Cauba
- Kendrick Calvo
- Nathanael Martinez
- Hannah Tuazon