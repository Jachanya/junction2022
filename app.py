from flask import Flask
import customer_behaviour as cb
import model as model

checkpoint_filepath = './checkpoint'
model = model.user_behaviour(27).load_weights(checkpoint_filepath)

app = Flask(__name__)

@app.route("/")
def index():
    return {"hello": "world"}

@app.route("/health/<user_id>")
def health(user_id):
    lab_out, out = cb.parse_user_id(user_id)
    outputs = []
    mod_out = model(out)

    index = tf.math.argmax(mod_out, axis = -1)
    action = cb.itoa[index]
    outputs.append(action)

    for i in range(1):
        out = tf.concat([out, mod_out], axis = 1)
        mod_out = model(out)
        index = tf.math.argmax(mod_out, axis = -1)
        action = cb.itoa[index]
        outputs.append(action)

    return {"prediction": output}

if __name__ == "__main__":
    app.run(port = 8000, debug = True)