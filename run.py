from app import create_app
import sys
import os
sys.path.append(os.path.dirname(__file__))


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)