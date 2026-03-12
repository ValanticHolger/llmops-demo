# Berrybrain MLflow Project

## Overview

This project utilizes MLflow for tracking experiments and managing machine learning models. It integrates with Hugging Face's OpenAI models to perform various natural language processing tasks.

## Requirements

- Python 3.x
- Pip
- Virtualenv

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/berrybrain.git
   cd berrybrain/mlflow
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install --upgrade pip
   pip install --no-cache-dir "mlflow[extras]"
   pip install python-dotenv openai
   ```

4. Start the MLflow server:
   ```bash
   ./start-mlflow.sh
   ```

## Usage

- Set your Hugging Face token in a `.env` file:
  ```
  HF_TOKEN=your_hugging_face_token
  ```

- Run the application:
  ```bash
  python app.py
  ```

- Access the MLflow UI at [http://127.0.0.1:5000](http://127.0.0.1:5000) to track your experiments.

## License

This project is licensed under the MIT License.