import os

from flask import Flask, request, render_template
from classifier.inference import judge_image
from flapi import ph_app

ph_app = Flask(__name__)


@ph_app.route("/", methods=["POST", "GET"])
def check_phone_model():
    '''
    Flask API for phone model checking.
    Asks for a photo of a model, returns True/False.
    :return: bool
    TODO: cool template for output
    '''
    print(os.getcwd())
    args = {"method": "GET"}
    if request.method == "POST":
        file = request.files["file"]
        result = judge_image(file)
        args["method"] = "POST"
        args["result"] = result
        args["image"] = file
    return render_template("check.html", args=args)

