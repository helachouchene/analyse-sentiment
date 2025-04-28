FROM tensorflow/tensorflow:2.10.0-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/app/static/uploads

CMD ["gunicorn", "-b", "0.0.0.0:8080", "app.app:app"]