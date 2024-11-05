from flask import Flask, request, jsonify
from flask_cors import CORS  
import openai

app = Flask(__name__)
CORS(app)  

openai.api_key = ""

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "A pergunta não pode estar vazia."}), 400

    # Formatação do prompt com instruções adicionais
    messages = [
        {"role": "system", "content": "Você é um especialista em economia e investimentos que trabalha em uma empresa ficticia chamada Solutec. Responda a pergunta de forma simples e direta, sem jargões complexos."},
        {"role": "system", "content": "e quando for perguntas referentes a valor de bitcoins mande a pessoa entrar na parte previsao  de valores de nosso site "},
         {"role": "system", "content": "lebre que vc tecnicamente trabalha em uma empresa do ramo economico chamada solutec "},
        {"role": "user", "content": question}
    ]

    try:
        # Chama a nova interface
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  
            messages=messages,
            max_tokens=150, 
            temperature=0.7,
        )
        answer = response.choices[0].message['content'].strip()
        
        # Retornar apenas a resposta, sem formatação adicional
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)