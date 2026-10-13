import requests
from flask import (
    Flask, render_template, request,
    session, redirect, url_for, flash
)
from functools import wraps
from werkzeug.security import (
    check_password_hash, generate_password_hash
)
import os
import sqlite3
from datetime import datetime
import pandas as pd
import joblib
import requests

app = Flask(__name__)

# Secret key for login sessions
app.secret_key = os.environ.get("SECRET_KEY")

if not app.secret_key:
    if os.environ.get("RENDER"):
        raise RuntimeError(
            "Please set SECRET_KEY in Render environment variables."
        )
    app.secret_key = os.urandom(32)

# Project files
DATA_FILE = "Crop_recommendation.csv"
MODEL_FILE = "crop_model.joblib"
DB_FILE = os.environ.get("DB_FILE", "predictions.db")

FEATURES = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]


# ---------------- DATABASE ----------------

def db():
    connection = sqlite3.connect(DB_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with db() as connection:

        # User login table
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        """)

        # Crop prediction history table
        connection.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                N REAL,
                P REAL,
                K REAL,
                temperature REAL,
                humidity REAL,
                ph REAL,
                rainfall REAL,
                crop TEXT
            )
        """)

        # Optional: create initial account using environment variables.
        # Set ADMIN_USERNAME and ADMIN_PASSWORD in Render.
        admin_username = os.environ.get(
            "ADMIN_USERNAME", ""
        ).strip()

        admin_password = os.environ.get(
            "ADMIN_PASSWORD", ""
        )

        if admin_username and admin_password:
            existing = connection.execute(
                "SELECT id FROM users WHERE username = ?",
                (admin_username,)
            ).fetchone()

            if not existing:
                connection.execute(
                    """
                    INSERT INTO users (username, password_hash)
                    VALUES (?, ?)
                    """,
                    (
                        admin_username,
                        generate_password_hash(admin_password)
                    )
                )


# ---------------- MACHINE LEARNING MODEL ----------------

def ensure_model():
    if os.path.exists(MODEL_FILE):
        return

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            "Crop_recommendation.csv must be in the same folder as app.py."
        )

    from sklearn.ensemble import RandomForestClassifier

    dataset = pd.read_csv(DATA_FILE)

    required_columns = FEATURES + ["label"]
    missing_columns = [
        column for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    model_object = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model_object.fit(
        dataset[FEATURES],
        dataset["label"]
    )

    joblib.dump(model_object, MODEL_FILE)


# ---------------- LOGIN PROTECTION ----------------

def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))

        return view(*args, **kwargs)

    return wrapped_view


# ---------------- USER REGISTRATION ----------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not password:
            flash("Please fill in all fields.", "error")
            return render_template("register.html")

        if len(username) < 3:
            flash("Username must be at least 3 characters.", "error")
            return render_template("register.html")

        if len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("register.html")

        try:
            with db() as connection:
                connection.execute(
                    """
                    INSERT INTO users (username, password_hash)
                    VALUES (?, ?)
                    """,
                    (username, generate_password_hash(password))
                )

            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            flash("Username already exists. Choose another.", "error")

    return render_template("register.html")

# ---------------- LOGIN PAGE ----------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "POST":

        username = request.form.get(
            "username", ""
        ).strip()

        password = request.form.get(
            "password", ""
        )

        with db() as connection:
            user = connection.execute(
                """
                SELECT id, username, password_hash
                FROM users
                WHERE username = ?
                """,
                (username,)
            ).fetchone()

        if user and check_password_hash(
            user["password_hash"], password
        ):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]

            return redirect(url_for("home"))

        flash("Invalid username or password.", "error")

    return render_template("login.html")


# ---------------- LOGOUT ----------------

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------- DASHBOARD DATA ----------------

def dashboard_data():

    if not os.path.exists(DATA_FILE):
        return {
            "crop_labels": [],
            "crop_counts": [],
            "avg_temp": 0,
            "avg_humidity": 0,
            "avg_rainfall": 0,
            "avg_ph": 0
        }

    dataset = pd.read_csv(DATA_FILE)

    counts = dataset["label"].value_counts().head(12)

    return {
        "crop_labels": counts.index.tolist(),
        "crop_counts": counts.values.tolist(),
        "avg_temp": round(
            float(dataset["temperature"].mean()), 1
        ),
        "avg_humidity": round(
            float(dataset["humidity"].mean()), 1
        ),
        "avg_rainfall": round(
            float(dataset["rainfall"].mean()), 1
        ),
        "avg_ph": round(
            float(dataset["ph"].mean()), 2
        )
    }


# ---------------- CROP PREDICTION PAGE ----------------

@app.route("/", methods=["GET", "POST"])
@login_required
def home():

    prediction = None
    error = None
    submitted = {}

    if request.method == "POST":

        try:
            submitted = {
                key: request.form.get(key, "").strip()
                for key in FEATURES
            }

            values = {
                key: float(submitted[key])
                for key in FEATURES
            }

            input_data = pd.DataFrame(
                [[values[key] for key in FEATURES]],
                columns=FEATURES
            )

            prediction = str(
                model.predict(input_data)[0]
            )

            # Save prediction to SQLite history
            with db() as connection:
                connection.execute(
                    """
                    INSERT INTO history (
                        created_at, N, P, K,
                        temperature, humidity, ph,
                        rainfall, crop
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        values["N"],
                        values["P"],
                        values["K"],
                        values["temperature"],
                        values["humidity"],
                        values["ph"],
                        values["rainfall"],
                        prediction
                    )
                )

        except (ValueError, TypeError, KeyError):
            error = (
                "Please enter a valid number in every field."
            )

    with db() as connection:

        recent_history = connection.execute(
            """
            SELECT * FROM history
            ORDER BY id DESC
            LIMIT 8
            """
        ).fetchall()

        total_predictions = connection.execute(
            "SELECT COUNT(*) AS total FROM history"
        ).fetchone()["total"]

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        submitted=submitted,
        history=recent_history,
        total_predictions=total_predictions,
        dashboard=dashboard_data()
    )


# ---------------- WEATHER FUNCTION ----------------

@app.route("/api/weather")
@login_required
def api_weather():
    city = request.args.get("city", "").strip()

    if not city:
        return {"error": "Please enter a city."}, 400

    result = get_weather(city)
    return result


def get_weather(city):

    try:
        # Find city coordinates
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json"
            },
            timeout=15
        )

        geo_response.raise_for_status()

        results = geo_response.json().get(
            "results", []
        )

        if not results:
            return {
                "error": (
                    f"Could not find '{city}'. "
                    "Please try another city."
                )
            }

        place = results[0]

        # Get current weather
        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": (
                    "temperature_2m,relative_humidity_2m,"
                    "precipitation,rain,wind_speed_10m"
                ),
                "timezone": "auto"
            },
            timeout=15
        )

        weather_response.raise_for_status()

        current = weather_response.json().get(
            "current", {}
        )

        if not current:
            return {
                "error": "Weather data is temporarily unavailable."
            }

        return {
            "city": (
                f'{place["name"]}, '
                f'{place.get("country", "")}'
            ),
            "temperature": current.get(
                "temperature_2m", "—"
            ),
            "humidity": current.get(
                "relative_humidity_2m", "—"
            ),
            "rain": current.get(
                "rain",
                current.get("precipitation", "—")
            ),
            "wind": current.get(
                "wind_speed_10m", "—"
            ),
            "time": current.get("time", "")
        }

    except requests.RequestException as error:
        app.logger.warning(
            "Weather API request failed: %s", error
        )

        return {
            "error": (
                "Weather service unavailable. "
                "Please try again later."
            )
        }

    except (ValueError, KeyError, TypeError) as error:
        app.logger.warning(
            "Weather response error: %s", error
        )

        return {
            "error": "Invalid weather data received."
        }
        





# ---------------- WEATHER PAGE ----------------


@app.route("/weather", methods=["GET", "POST"])
@login_required
def weather():
    city = request.values.get("city", "").strip()
    weather_result = None

    if city:
        weather_result = get_weather(city)

    return render_template(
        "weather.html",
        city=city,
        weather=weather_result
    )


# ---------------- HISTORY PAGE ----------------

@app.route("/history")
@login_required
def history_page():

    with db() as connection:
        rows = connection.execute(
            """
            SELECT * FROM history
            ORDER BY id DESC
            LIMIT 100
            """
        ).fetchall()

    return render_template(
        "history.html",
        history=rows
    )


# ---------------- START APPLICATION ----------------

init_db()
ensure_model()
model = joblib.load(MODEL_FILE)

# Weather API: city name to weather details



if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
