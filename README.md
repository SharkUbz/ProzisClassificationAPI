# Prozis Intent Classifier API

This REST API classify inserted sentences by users with 6 predefined intentions (search_product, product_info, add_to_cart, go_to_shopping_cart, confirm_order, unknown_intention).

## Stack
- Python 3.9
- FastAPI
- scikit-learn (TF-IDF + MLP or any classifier from the package)
- Uvicorn

## How To Execute Locally

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Training Model(s) (You need to change the location where the file is saved and then loaded)
```bash
python app/train_model.py
```

### 3. Executar a API
```bash
uvicorn app.main:app --reload
```

### 4. Test with `Swagger`
```bash
Go to http://127.0.0.1:8000/docs

Click in POST /classify and then in Try it out

Write the wanted text and press Execute
```

### 5. Test with `Postman`
```bash
Click New, HTTP Request

Set method to POST

Set URL to http://127.0.0.1:8000/classify

Go to the Body tab

Select raw and choose JSON

Write the wanted text and press Send
```

## Output Example (barra de proteínas)
```json
{
  "intent": "search_product",
  "confidence_score": 0.93
}
```

## Explicações
A linguagem que utilizei foi o `Python`. Como mencionei na entrevista a minha experiência com Inteligência Artificial 
foi muito focada em Python, que também acaba por ser uma linguagem mais simples, legível e tem disponíveis muitas bibliotecas que
podem ser utilizadas para a implementação destes sistemas.

Como nunca tinha trabalhado com APIs em Python decidi experimentar o `FastAPI`, uma vez tem um desempenho elevado
e suporte automático.

- Escolhi uma abordagem simples onde fiz:

- Vetorização com `TfidfVectorizer` para representar frases em forma numérica.

Testei os classificadores do pacote de `scikit-learn`. A decisão de não usar modelos mais complexos com ferramentas como 
o Tensorflow e Transformers (ex: redes neuronais mais avançadas e complexas, BERT, etc.) foi tomada tendo em conta que:

- O dataset fornecido era pequeno e bem definido, não justificando o custo/complexidade de treinar modelos profundos.

- O foco foi entregar uma solução leve, funcional e extensível, com tempo de resposta rápido e baixa dependência computacional.

Em contextos reais, a abordagem pode ser evoluída incrementalmente com modelos mais sofisticados ou embeddings sem 
alterar a arquitetura da API.

Se estivessemos a falar num contexto onde o utilizador interagiria com um chatbot, tendo em conta por exemplo o que eu 
desenvolvi no passado, os dados de texto teriam de ser mais robustos.

Vamos utilizar por exemplo um input do utilizador a interagir com o chatbot,
o utlizador tem em mente o que procura, porém ele não sabe a terminologia correta do produto que está à procura e tenta
explicar ao chatbot os detalhes do produtos, por exemplo, neste caso o modelo que eu criaria para este tipo de situações
teria um normalizador de texto que envolveria expandir or texto se tivesse concatenações, converteria o texto para letras
minúsculas, lematização para ter a raíz da palavra e remoção de stop words sem remoção das negativas que podem contribuír
para uma procura mais específica (ex: não tem cálcio).




## Melhorias Futuras
- Criar mais dummy data ou colocar mais dados nos dados que me foram fornecidos;
- Mais uma vez, o uso de ferramentas mais complexas.
