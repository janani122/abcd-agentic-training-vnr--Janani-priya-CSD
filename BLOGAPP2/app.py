from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Store blog posts in a list of dictionaries
posts = [
    {"title": "Welcome to BlogBox", "content": "This is the first post in our blog."},
    {"title": "Another Post", "content": "Here's some more content."}
]

@app.route('/')
def home():
    return render_template('home.html', posts=posts)

@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        title = request.form['title']
        content = request.form['content']
        posts.append({"title": title, "content": content})
        return redirect(url_for('home'))
    return render_template('add.html')

if __name__ == '__main__':
    app.run(debug=True)