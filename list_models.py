import os
import google.generativeai as genai

api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyDqLC6HYhkAVcAkm86CZwbq7vPUxSNAJg8").strip()
genai.configure(api_key=api_key)

try:
    print("Listing available models:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")
