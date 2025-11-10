# 💻🔧Project Setup

### 📱 **Project Overview:**
This thesis project, developed by Computer Science students from the University of Santo Tomas, introduces a beta-stage system designed to streamline defense demonstrations through a user-friendly interface. Users can upload an image and choose between classification modes—with or without normalization. The system then generates Grad-CAM visualizations for each model, along with their respective classification results and confidence scores.

----

### 🩴 **Steps:**
1. Open separate terminals for /backend and /frontend
2. Download the models in [Models Link](https://drive.google.com/drive/folders/1ODgIc6DfYfZssT6wOi9b5Y6oQXxM0Q5g?usp=drive_link)
3. Inside the `/backend` folder place the models to its corresponding folders (e.g., backend/models/coatnet/coatnet_normalized.pth).
2. **For Backend:**
``` bash
pip install -r requirements.txt
python run.py   # To run the backend server
```
3. **For Frontend:**
``` bash
npm install
npm run dev     # To launch web application
```
4. Open browser and enter `localhost:5173`.
5. Upload an image (Margin Negative or Margin Positive)
6. Select classification type *w/ Normalization* or *w/o Normalization*.

----

### 👨‍👩‍👧‍👦 **Authors:**
- David Raniel Cauba
- Kendrick Calvo
- Nathanael Martinez
- Hannah Tuazon