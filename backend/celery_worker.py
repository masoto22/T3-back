from celery import Celery
from process_scripts import process_scripts

app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task
def run_process_scripts():
    process_scripts()