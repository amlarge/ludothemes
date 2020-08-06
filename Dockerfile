FROM python:3.7.4
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 500
ENTRYPOINT ["python"]
CMD ["app.py"]
