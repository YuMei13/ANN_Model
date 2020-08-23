from flask import Flask
from flask import render_template
import dash
flask_app = Flask(__name__)

@flask_app.route('/')
def index():
    #return 'Home Page<a href="/coffee_machine">Coffee Machine Prediction Model</a>'

    return render_template('flask_home.html')

'''

if __name__ == '__main__':
    flask_app.run()
'''
