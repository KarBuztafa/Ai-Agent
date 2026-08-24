import os
from flask import Flask, request, jsonify, render_template
from groq import Groq
from google import genai

app = Flask(__name__)

# API Bağlantıları
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

def execute_patron_calisan_flow(user_request):
    # 1. ADIM: Groq (Patron) isteği analiz eder ve Gemini için net teknik direktif hazırlar
    patron_directive_prompt = f"""
    Sen sistemin PATRON ajansısın. Kullanıcının şu isteğini incele ve ÇALIŞAN ajanın (Gemini) 
    hatasız uygulayabilmesi için net, teknik ve adımsayarla yazılmış bir prompt (direktif) oluştur.
    Kullanıcı İsteği: {user_request}
    """
    
    patron_directive = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": patron_directive_prompt}]
    ).choices[0].message.content

    # 2. ADIM: Gemini (Çalışan) patronun direktifini uygular ve ürünü hazırlar
    gemini_work = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=patron_directive
    ).text

    # 3. ADIM: Groq (Patron) çalışan çıktısını denetler (Review & QA)
    patron_review_prompt = f"""
    Sen denetleyici PATRON ajansın. 
    Kullanıcı İsteği: {user_request}
    Çalışanın Ürettiği Taslak:
    {gemini_work}

    GÖREVİN:
    1. Taslakta mantık hatası, eksik veya kod kusuru var mı kontrol et.
    2. Eğer taslak zaten mükemmelse HİÇBİR YORUM EKLEMEDEN doğrudan cevabı kullanıcıya sun.
    3. Eğer hata varsa hızlıca düzeltip en mükemmel ve nihai haliyle cevap ver.
    """

    final_verified_response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": patron_review_prompt}]
    ).choices[0].message.content

    return final_verified_response

@app.route("/process", methods=["POST"])
def process():
    data = request.get_json()
    task = data.get("task", "")
    
    if not task:
        return jsonify({"error": "Görev belirtilmedi"}), 400
        
    result = execute_patron_calisan_flow(task)
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run()
