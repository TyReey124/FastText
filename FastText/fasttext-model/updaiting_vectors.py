import fasttext
import numpy as np
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pymorphy3
import os

# Загрузка необходимых ресурсов
nltk.data.path.append(os.path.abspath("./nltk_data"))
nltk.download("punkt", download_dir="./nltk_data")
nltk.download("stopwords", download_dir="./nltk_data")
morph = pymorphy3.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))

# Подгрузка модели
model = fasttext.load_model("./fasttext-model/cc.ru.300.bin")

# Загрузка title из json
json_file_path = './data-base/dataset.json'
with open(json_file_path, 'r', encoding='utf-8') as file:
    dataset = json.load(file)
questions = [item['title'] for item in dataset['data']]

# Лемматизация текста и приведение к нижнему регистру
def lemmatize_text(text):
    text = text.lower()
    words = word_tokenize(text, language="russian")
    words = [word for word in words if word.isalnum() and word not in stop_words]
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmatized_words)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Цикл для лемматизации каждого вопроса
lemmatized_questions = []
for question in questions:
    lemmatized_question = lemmatize_text(question)
    lemmatized_questions.append({"question": question, "lemmatized_question": lemmatized_question})
questions = [item["lemmatized_question"] for item in lemmatized_questions]

# Превращение title из dataset в вектора через FastText
def sentence_to_vector(sentence, model):
    words = sentence.split()
    word_vectors = [model.get_word_vector(word) for word in words if word in model]
    sentence_vector = np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.get_dimension())
    return sentence_vector
question_vectors = np.array([sentence_to_vector(q, model) for q in questions])

# запись весов в json file
json_file_path = './fasttext-model/dataset-vectors/fasttext_weights.json'
question_vectors_list = question_vectors.tolist()
save_json(question_vectors_list, json_file_path)

print(questions)