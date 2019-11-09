#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Nov 8, 2019

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Flask-based RESTful API for KBQA on DBpedia
'''
from flask import Flask, jsonify, request
from request import KBQA

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello, World!"


@app.route('/ask', methods=['GET'])
def ask_qamp():
    question = request.args.get('question', type=str)
    print(question)
    service = KBQA()
    answers = service.request(question, verbose=False)
    return jsonify({'answers': answers})


if __name__ == '__main__':
    app.run(debug=True)
