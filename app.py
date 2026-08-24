import os
from flask import Flask, request, jsonify, render_template
from groq import Groq
from google import genai

app = Flask(__name__)

# API Bağlantıları (Render'daki değişken isimlerinle güncellendi)
groq_client = Groq(api_key=os.environ.get("groq_apikey"))
gemini_client = genai.Client(api_key=os.environ.get("gemini_apikey"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    data = request.get_json()
    user_request = data.get("task", "")

    if not user_request:
        return jsonify({"error": "Lütfen bir görev girin."}), 400

    try:
        # 1. ADIM: Groq (Patron) direktif oluşturur
        patron_prompt = f"Sen PATRON ajansın. Kullanıcının şu isteği için Gemini'a verilecek net direktifi hazırla: {user_request}"
        patron_directive = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": patron_prompt}]
        ).choices[0].message.content

        # 2. ADIM: Gemini (Çalışan) taslağı üretir
        gemini_work = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=patron_directive
        ).text

        # 3. ADIM: Groq (Patron) taslağı denetler ve son hali verir
        review_prompt = f"Kullanıcı İsteği: {user_request}\nÇalışan Taslağı: {gemini_work}\n\nBu taslağı incele, varsa hataları düzelt ve kullanıcıya mükemmel, net yanıtı ver."
        final_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": review_prompt}]
        ).choices[0].message.content

        return jsonify({"result": final_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
    
