import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pymorphy3
import model

# Загрузка необходимых ресурсов
nltk.download("punkt")
nltk.download("stopwords")
morph = pymorphy3.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))

# Загрузка модели fasttext
model = fasttext.load_model("./fasttext-model/cc.ru.300.bin")

# Загрузка данных из dataset
with open('./data-base/dataset.json', 'r', encoding='utf-8') as file:
    dataset_json = json.load(file)
questions = [item['title'] for item in dataset_json['data']]

# Лемматизация и унификация слов
def lemmatize_text(text):
    text = text.lower()
    words = word_tokenize(text, language="russian")
    words = [word for word in words if word.isalnum() and word not in stop_words]
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    return ' '.join(lemmatized_words)
lemmatized_questions = [{'question': question, 'lemmatized_question': lemmatize_text(question)} for question in questions]
questions = [item["lemmatized_question"] for item in lemmatized_questions]

# Перевод предложения в вектор fasttext
def sentence_to_vector(sentence, model):
    words = sentence.split()
    word_vectors = [model.get_word_vector(word) for word in words if word in model]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.get_dimension())


# Загрузка векторов всех title из json
with open('./fasttext-model/dataset-vectors/fasttext_weights.json', 'r', encoding='utf-8') as file:
    vector_data = json.load(file)

# Нахождение похожих векторовq
question_vectors = np.array(vector_data)
def find_top_similar_questions(new_question, question_vectors, questions, model):
    new_vector = sentence_to_vector(new_question, model)
    similarities = cosine_similarity([new_vector], question_vectors)[0]
    top_indices = np.argsort(similarities)[::-1]
    top_questions = []
    for idx in top_indices:
        similarity = similarities[idx]
        if similarity >= 0.5:
            if top_questions and (top_questions[0][1] - similarity) > 0.01:
                break
            top_questions.append((questions[idx], similarity))
        if len(top_questions) >= 4:
            break
    return top_questions

# нахождение вопросов из датасета, соответсутвующих найденным вопросам, которые лемматизированы
def find_original_question(similar_question, dataset):
    for item in dataset['data']:
        if similar_question == lemmatize_text(item['title']):
            return item
    return None

# Получение по исходному title description и url
def get_answers_with_details(top_similar_questions, dataset):
    answers_with_details = []
    for question, similarity in top_similar_questions:
        original_question_data = find_original_question(question, dataset)
        if original_question_data:
            answer_detail = {
                "title": original_question_data["title"],
                "description": original_question_data["description"],
                "url": original_question_data["url"]
            }
            answers_with_details.append(answer_detail)
    return answers_with_details

# загрузка классифицированных данных в json для передачи в liama
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

json_file_path = './querry/prompt.json'

# Вызов
def answer(user_question):
    user_question = lemmatize_text(user_question)
    top_similar_questions = find_top_similar_questions(user_question, question_vectors, questions, model)
    answers_with_details = get_answers_with_details(top_similar_questions, dataset_json)

    prompt = answers_with_details if answers_with_details else [{"answer": "Извините, я не могу найти ответ на ваш вопрос."}]
    save_json(prompt, json_file_path)
    
    return model.generate_answer(prompt)