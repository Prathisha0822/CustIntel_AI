# CustIntel - AI-Powered Customer Intelligence Platform 🚀

## Overview
CustIntel is a comprehensive E-commerce dashboard and AI command center. It transforms raw dataset files into actionable business insights using Classical Machine Learning (LTV & Churn Prediction), PyTorch Deep Learning (Next Purchase Prediction), and Google Gemini Generative AI (Text-to-SQL & Review Summarization).

## Tech Stack
* **Frontend UI:** Streamlit
* **Database:** SQLite & Pandas
* **Machine Learning:** Scikit-Learn (Random Forest, KNN, Linear Regression)
* **Deep Learning:** PyTorch (MLP Neural Networks)
* **Generative AI:** Google Gemini LLM API

## Quick Start

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
2. **Set Environment Variables: Make sure you have your .env file set up in the root directory:**
GEMINI_API_KEY=your_key_here

3.**Build the Database & Train Models: After running the python files the model gets generated**
python src/data_ingestion.py
python src/ml_models.py
python src/dl_model.py

4. **Launch the Dashboard:**
streamlit run app.py

*Customer-Dashboard*
<img width="1903" height="1017" alt="image" src="https://github.com/user-attachments/assets/5695d459-0760-4913-9e40-64e3b93e3cc4" />

**Chrun & LTV Prediction System**
<img width="1540" height="876" alt="image" src="https://github.com/user-attachments/assets/b0c3d96a-d2b9-4945-93cb-adbfe2a34d27" />

**Recomendation-System**
<img width="1532" height="999" alt="image" src="https://github.com/user-attachments/assets/889e402d-9f32-4121-84c4-c96fe12af934" />

**Chadbot**
<img width="1533" height="996" alt="image" src="https://github.com/user-attachments/assets/4eab0ed0-e7aa-4afd-acd7-13dd62ce6e7a" />



