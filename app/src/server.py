#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
# @Time    : 8/3/2021 5:21 PM
# @Author  : sanket.tijare@icertis.com
# @File    : server.py
"""
import os
from flask import Flask, request
from flask import render_template
from predict import predict
import shutil

app = Flask(__name__)
UPLOAD_FOLDER = './static/uploaded_images/'


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        sku_type = request.values['sku_type']
        print(sku_type)
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict(image_location, sku_type)
            op = []
            if pred:
                for f in pred:
                    shutil.copy2(f['filename'], './static/show_images/')
                    fname = "./static/show_images/" + f['filename'].split('/')[-1]
                    op.append({'filename': fname, 'similarity':f['similarity']})

                result = []
                for x in range(4):
                    try:
                        result.append(op[x])
                    except:
                        result.append({'similarity': None, 'filename': None})

                return render_template("index.html", prediction=result, image_loc=result, input_image=image_location,
                                       nothing_found=None)
            else:
                print("Nothing Found")
                return render_template("index.html", prediction=None, image_loc=None, input_image=None,
                                       nothing_found='nothing_found')
    return render_template("index.html", prediction=None, image_loc=None, input_image=None, nothing_found=None)


if __name__ == '__main__':
    app.run(port=8080, debug=True)