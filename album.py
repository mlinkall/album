from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import socket
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # max file size is 1GB
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("file[]")
        filenames = []
        today = datatime.now()
        date_folder_name = today.strftime("%Y%m%d")
        target_directory_path = os.path.join(app.config['UPLOAD_FOLDER'], data_folder_name)
        
        try:
            os.makedirs(target_directory_path, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {target_directory_path}: {e}")
            return "Server error: Could not create upload directory.", 500
        
        filenames = []
        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                #filenames.append(filename)
                file_save_path = os.path.join(target_directory_path, filename)
                file.save(file_save_path)
                
                relative_path_for_link = os.path.join(data_folder_name, filename)
                filenames.append(relative_path_for_link)
                
        return render_template('upload.html', filenames=filenames)
        
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename) 
    
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()
    finally:
        s.close()
    return local_ip

if __name__ == '__main__':
   print(f"Local IP address: {get_local_ip()}")
   app.run(host='0.0.0.0',port=8080,debug=True)
   


