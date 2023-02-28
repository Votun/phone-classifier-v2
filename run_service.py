from flapi import ph_app
from classifier.inference import ConvBlock, SimpleCNN
ph_app.ph_app.run(host='0.0.0.0', debug=True)
