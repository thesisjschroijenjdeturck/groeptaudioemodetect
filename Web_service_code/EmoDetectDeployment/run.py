from predictormanager import app
from flask import Response
from flask import jsonify
from flask_restful import Resource, Api

if __name__ == "__main__":
    app.run( debug = True )
