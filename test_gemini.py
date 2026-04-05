import os
import google.generativeai as genai

api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyDqLC6HYhkAVcAkm86CZwbq7vPUxSNAJg8").strip()
genai.configure(api_key=api_key)

models_to_test = ["gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro", "gemini-pro"]

for m_name in models_to_test:
    print(f"Testing {m_name}...")
    try:
        model = genai.GenerativeModel(m_name)
        response = model.generate_content("Hello")
        if response and response.text:
            print(f"  SUCCESS! {m_name}")
            break
    except Exception as e:
        print(f"  FAILED: {m_name} - {e}")
