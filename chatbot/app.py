from flask import Flask, request, jsonify
import nltk
from nltk.chat.util import Chat, reflections

nltk.download("punkt")

app = Flask(__name__)

# Padrões de conversação
pairs = [
    [
        r"(Oi|Olá|E aí)",
        ["Olá! Como posso te ajudar?", "Oi! Em que posso ser útil?"],
    ],
    [
        r"(Como comprar ações?|Como investir em ações?|Onde comprar ações?)",
        ["Você pode comprar ações através de uma corretora de valores. Primeiro, abra uma conta em uma corretora, deposite fundos e escolha as ações que deseja comprar."],
    ],
    [
        r"(Quais são os riscos de investir em ações?|Riscos de ações|Investir em ações é arriscado?)",
        ["Investir em ações pode ser arriscado devido à volatilidade do mercado. Os preços das ações podem subir ou descer rapidamente, e você pode perder parte ou todo o seu investimento."],
    ],
    [
        r"(Quais são os melhores ETFs?|Melhores ETFs para investir|Recomendações de ETFs)",
        ["Os melhores ETFs dependem dos seus objetivos de investimento. Alguns populares incluem o SPY, que rastreia o S&P 500, e o QQQ, que rastreia o Nasdaq-100."],
    ],
    [
        r"(Quais são os benefícios dos ETFs?|Vantagens dos ETFs|Por que investir em ETFs?)",
        ["ETFs oferecem diversificação, baixos custos e facilidade de negociação, pois são comprados e vendidos como ações na bolsa de valores."],
    ],
    [
        r"(Como comprar criptomoedas?|Onde comprar criptomoedas?|Como investir em criptomoedas?)",
        ["Você pode comprar criptomoedas em exchanges como Coinbase, Binance e Kraken. Primeiro, crie uma conta, deposite fundos e escolha as criptomoedas que deseja comprar."],
    ],
    [
        r"(Quais são os riscos de investir em criptomoedas?|Riscos das criptomoedas|Investir em criptomoedas é arriscado?)",
        ["Criptomoedas são altamente voláteis e podem apresentar grandes oscilações de preço. Além disso, existem riscos de segurança e regulamentação."],
    ],
    [
        r"(Quais são os melhores investimentos em renda fixa?|Melhores opções de renda fixa|Recomendações de renda fixa)",
        ["Algumas opções populares de renda fixa incluem títulos do governo, CDBs, LCIs e LCAs. A escolha depende do seu perfil de risco e objetivos financeiros."],
    ],
    [
        r"(Quais são os riscos da renda fixa?|Riscos de investir em renda fixa|Investir em renda fixa é seguro?)",
        ["Investimentos em renda fixa são geralmente considerados mais seguros do que ações, mas ainda existem riscos, como risco de crédito e risco de inflação."],
    ],
    [
        r"(Como começar a investir?|Dicas para iniciantes em investimentos|Por onde começar a investir?)",
        ["Para começar a investir, defina seus objetivos financeiros, crie um orçamento, escolha uma corretora e comece com investimentos básicos, como renda fixa ou ETFs."],
    ],
    [
        r"(Qual é o melhor investimento para iniciantes?|Melhores investimentos para iniciantes|Onde investir como iniciante?)",
        ["ETFs e fundos de investimento são boas opções para iniciantes devido à sua diversificação e gestão profissional. Renda fixa também é uma boa escolha para menor risco."],
    ],
    [
        r"(O que é planejamento financeiro?|Como fazer um planejamento financeiro?|Importância do planejamento financeiro)",
        ["Planejamento financeiro envolve definir objetivos financeiros, criar um orçamento, economizar dinheiro e investir de forma inteligente para alcançar seus objetivos."],
    ],
    [
        r"(Como criar um orçamento?|Dicas para criar um orçamento|Importância de um orçamento)",
        ["Para criar um orçamento, liste suas receitas e despesas mensais, identifique áreas onde você pode economizar e defina metas de economia e investimento."],
    ],
    [
        r"(O que são ações?|Como funcionam as ações?|Investimento em ações)",
        ["Ações representam uma parte do capital social de uma empresa. Ao comprar ações, você se torna um acionista e participa dos lucros e prejuízos da empresa."],
    ],
    [
        r"(O que é renda fixa?|Como funciona renda fixa?|Investimento em renda fixa)",
        ["Renda fixa é um tipo de investimento onde o investidor empresta seu dinheiro a uma instituição financeira, governo ou empresa em troca de uma remuneração fixa, como juros."],
    ],
    [
        r"(O que é renda variável?|Como funciona a renda variável?|Investimento em renda variável)",
        ["Renda variável é um tipo de investimento onde o retorno não é previsível, pois depende do desempenho do ativo no mercado. Exemplos incluem ações, ETFs e criptomoedas."],
    ],
    [
        r"(O que são ETFs?|Como funcionam os ETFs?|Investimento em ETFs)",
        ["ETFs (Exchange Traded Funds) são fundos que replicam a performance de um índice, como o S&P 500. Eles são negociados na bolsa de valores como ações."],
    ],
    [
        r"(O que são criptomoedas?|Como funcionam as criptomoedas?|Investimento em criptomoedas)",
        ["Criptomoedas são moedas digitais que utilizam criptografia para garantir transações seguras. Exemplos incluem Bitcoin, Ethereum e muitas outras."],
    ],
    [
        r"(.*ETF.*|.*etf.*|.*ETFs.*|.*etfs.*)",
        ["ETFs (Exchange Traded Funds) são fundos que replicam a performance de um índice, como o S&P 500. Eles são negociados na bolsa de valores como ações."],
    ],
    [
        r"(.*criptomoeda.*|.*criptomoedas.*|.*cripto.*)",
        ["Criptomoedas são moedas digitais que utilizam criptografia para garantir transações seguras. Exemplos incluem Bitcoin, Ethereum e muitas outras."],
    ],
    [
        r"(.*investimento.*|.*investimentos.*)",
        ["Como posso te ajudar mais especificamente com investimentos? Você tem alguma dúvida específica?"],
    ],
    [
        r"(.*renda fixa.*)",
        ["Renda fixa é um tipo de investimento onde o investidor empresta seu dinheiro a uma instituição financeira, governo ou empresa em troca de uma remuneração fixa, como juros. Posso te ajudar com mais alguma coisa?"],
    ],
    [
        r"(.*renda variável.*|renda variavel)",
        ["Renda variável é um tipo de investimento onde o retorno não é previsível, pois depende do desempenho do ativo no mercado. Exemplos incluem ações, ETFs e criptomoedas. Posso te ajudar com mais alguma coisa?"],
    ],
    [
        r"(ok.*|obrigado.*|tchau|até logo|adeus)",
        ["De nada! Se precisar de mais alguma coisa, estou à disposição. Até logo!"],
    ],
    [
        r"(.*)",
        ["Desculpe, eu não entendi. Você poderia reformular sua pergunta?", "Poderia explicar melhor?"]
    ]
]

# Inicialização do Chatbot
chatbot = Chat(pairs, reflections)

@app.route("/chat", methods=["POST"])
def chat():
    # Recebe a mensagem do cliente
    user_message = request.json.get("message")
    
    # Gera uma resposta a partir do chatbot
    bot_response = chatbot.respond(user_message)
    
    # Retorna a resposta como JSON
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
