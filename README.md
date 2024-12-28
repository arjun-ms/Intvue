# Intvue

**Intvue** is a web application designed to simulate technical interviews with an AI-powered virtual interviewer. Built using FastAPI for the backend and HTML/CSS for the frontend, Intvue helps users practice and ace their tech interviews.

---

## Features

- AI-powered technical interview simulations.
- Interactive user interface with responsive design.
- Backend built with FastAPI for high performance and scalability.
- Templates folder hosting HTML and CSS files for a seamless frontend experience.

---

## Technologies Used

- **FastAPI**: For building the backend API.
- **HTML/CSS**: For creating the user interface.
- **Python**: Backend programming language.

---

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment tool (e.g., `venv` or `virtualenv`)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/arjun-ms/Intvue.git
   cd https://github.com/arjun-ms/Intvue.git
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

5. Open your browser and visit:
   ```
   http://127.0.0.1:8000
   ```


---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push to your branch.
   ```bash
   git commit -m "Description of changes"
   git push origin feature-name
   ```
4. Open a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

