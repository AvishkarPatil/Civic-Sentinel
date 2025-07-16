# Civic Sentinel - Flask Implementation

This is the Flask implementation of the Civic Sentinel project, providing a more robust web application with user authentication, API endpoints, and a professional UI.

## Features

- **User Authentication** - Login and registration system
- **Interactive Dashboard** - Real-time analytics and visualization
- **REST API** - Programmatic access to the detection system
- **Responsive UI** - Bootstrap-based responsive design
- **Database Integration** - SQLite database for storing detection history

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Civic-Sentinel.git
   cd Civic-Sentinel
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_flask.txt
   ```

4. **Run the application**
   ```bash
   python run.py
   ```

5. **Open your browser** and navigate to `http://localhost:5000`

## Project Structure

```
Civic-Sentinel/
├── flask_app/
│   ├── __init__.py          # Application factory
│   ├── routes.py            # Main routes
│   ├── auth.py              # Authentication routes
│   ├── api.py               # API endpoints
│   ├── models.py            # Database models
│   ├── static/              # Static files (CSS, JS, images)
│   └── templates/           # HTML templates
├── anomaly_detector.py      # Core detection model
├── run.py                   # Application entry point
├── requirements_flask.txt   # Flask dependencies
└── README_FLASK.md          # This file
```

## API Usage

### Check API Status

```bash
curl http://localhost:5000/api/status
```

### Detect Anomalies

```bash
curl -X POST -F "file=@road.jpg" http://localhost:5000/api/detect
```

## User Authentication

The application includes a user authentication system. To access protected features like analytics and history:

1. Create an account using the Sign Up page
2. Log in with your credentials
3. Access protected routes

## Development

To run the application in development mode:

```bash
export FLASK_APP=run.py
export FLASK_ENV=development
flask run
```

## Deployment

For production deployment, consider using Gunicorn:

```bash
gunicorn -w 4 "run:app"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.