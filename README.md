# Project Title
# Streamlit App Setup and Troubleshooting Guide

## **Prerequisites**
Ensure you have Python installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/).

---

## **Step 1: Install Required Packages**
To set up your Streamlit app, install the required Python libraries. Use the following commands in your terminal or command prompt:

```bash
pip install streamlit PyPDF2 langchain faiss-cpu openai sentence-transformers tiktoken
pip install streamlit langchain openai tiktoken pdfplumber pandas Pillow
pip install pymupdf
```

### Verbose Logging (Optional)
If you suspect an error and want detailed logs, add the `-v` flag to the pip commands:

```bash
pip install -v streamlit PyPDF2 langchain faiss-cpu openai sentence-transformers tiktoken
```

---

## **Step 2: Activate Your Virtual Environment**
If you are using a virtual environment, activate it before proceeding:

### For Linux/macOS:
```bash
python -m venv venv
source venv/bin/activate
```

### For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

---

## **Step 3: Clear Cached Downloads (Optional)**
If pip is stuck or using a corrupted package, clear the cache:

```bash
pip cache purge
```

<!-- ![Pip Cache Purge Screenshot](./static/Readme.png) -->

---

## **Step 4: Generate Requirements File**
To ensure consistent environments, generate a `requirements.txt` file:

### For Windows:
```bash
pip freeze > requirements.txt
```

### For Linux/macOS:
```bash
pip install -r requirements.txt
```

![Requirements File Screenshot](./static/Readme.png)

---

## **Step 5: Run Your Streamlit App**
Finally, launch your Streamlit application:

```bash
streamlit run app.py
```

<!-- ![Streamlit App Running Screenshot](https://example.com/streamlit_app_running_screenshot.png) -->

---

## **Useful Links**
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Python Virtual Environment Guide](https://docs.python.org/3/library/venv.html)
- [Troubleshooting Pip](https://pip.pypa.io/en/stable/user_guide/#using-pip)

---

### **Need More Help?**
For further assistance, refer to the [Streamlit Community Forums](https://discuss.streamlit.io/).
