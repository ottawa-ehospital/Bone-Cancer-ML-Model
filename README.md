## Setup and Usage

1. **Installation**:

   Before you can use the API, you need to set up the project and install the necessary dependencies. You can do this using the following steps:

   - Install Python dependencies:

     ```bash
     pip install -r requirements.txt
     ```

2. **Run the API**:

   The FastAPI server can be started using the following command:

   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

   The API will be accessible at http://0.0.0.0:8000.