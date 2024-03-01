from flask import Flask,render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('3d_skeleton.html')
if __name__=="__main__":
    app.run(debug=True)