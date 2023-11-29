FROM python:3.8-slim

RUN pip install numpy pandas scikit-learn tabulate joblib

RUN mkdir my_model

ENV MODEL_DIR=/home/jovyan/my_model
ENV MODEL_NAME=best_extra_trees_model.pkl

COPY diagnostic.py /home/jovyan/my_model/diagnostic.py
COPY best_extra_trees_model.pkl /home/jovyan/my_model/best_extra_trees_model.pkl

CMD ["python", "/home/jovyan/my_model/diagnostic.py"]