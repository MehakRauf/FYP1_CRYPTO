FROM python:3.10

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the project
COPY . .

# Make sure .streamlit folder is writable
RUN mkdir -p /app/.streamlit

# Optional: disable usage stats
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
