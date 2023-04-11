FROM python:3.9.15-slim-buster
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y ca-certificates --no-install-recommends && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY text_Classification.py ./text_Classification.py
EXPOSE 8501
CMD ["streamlit", "run", "text_Classification.py"]
