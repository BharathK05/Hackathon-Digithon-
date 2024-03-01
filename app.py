from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
app.config['SECRET KEY'] = "ABCDEFG123345"
socketio = SocketIO(app)

# Load GPT2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
chatbot = GPT2LMHeadModel.from_pretrained("gpt2")
user_groups = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create_group', methods=['POST'])
def create_group():
    group_key = request.form['group_key']
    user_groups[group_key] = {'users': set()}
    return redirect(url_for('group_chat', group_key=group_key))

@app.route('/group_chat/<group_key>')
def group_chat(group_key):
    if group_key not in user_groups:
        return "Group Not Found"
    return render_template('group_chat.html', group_key=group_key)

@socketio.on('join_group')
def join_group(data):
    group_key = data['group_key']
    username = data['username']
    user_groups[group_key]['users'].add(username)
    emit('update_users', {'users': list(user_groups[group_key]['users'])}, room=group_key)

@socketio.on('leave_group')
def leave_group(data):
    group_key = data['group_key']
    username = data['username']
    user_groups[group_key]['users'].remove(username)
    emit('update_users', {'users': list(user_groups[group_key]['users'])}, room=group_key)

@socketio.on('chat_message')
def chat_message(data):
    group_key = data['group_key']
    username = data['username']
    message = data['message']

    # Use GPT2 to generate a response
    input_ids = tokenizer.encode(message, return_tensors="pt")
    response_ids = chatbot.generate(input_ids, max_length=150, temperature=0.7, num_return_sequences=1)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    emit('message', {'username': username, 'message': f'You: {message}', 'is_user': True}, room=group_key)
    emit('message', {'username': 'ChatBot', 'message': f'ChatBot: {response}', 'is_user': False}, room=group_key)

if __name__ == '__main__':
    socketio.run(app, debug=True)
