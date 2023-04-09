import streamlit as st
from transformers import pipeline

model = pipeline("question-answering",
                 model='deepset/roberta-base-squad2')


st.title('Question Answering')
hint_text = "Вывод"
description_text = "Модель находящая ответы на вопросы из контекста"
st.subheader(description_text)

text_question = st.text_input(label='Введите вопрос')
text_context = st.text_area(label='Введите контекст')
button = st.button('Получить ответ')


if button and text_question and text_context:
    response = model(text_question, text_context)
    if ('answer' in response):
        st.markdown(hint_text)
        st.write(f"""
                 Оценка: {response['score']}\n
                 Ответ: {response['answer']}\n
                 Ссылка для ознакомления с моделью: https://huggingface.co/deepset/roberta-base-squad2
                  """)
    else:
        st.markdown("Невозможно дать ответ", unsafe_allow_html=False)
else:
    st.markdown(hint_text)