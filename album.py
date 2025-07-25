from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import socket
from datetime import datetime
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # max file size is 1GB
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif', 'mp4', 'mov'])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])

app.logger.setLevel(logging.DEBUG)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
           
def get_dir_name():
    today = datetime.now()
    dir_name = today.strftime("%Y%m%d")
    app.logger.debug(f"Generated directory name: {dir_name}")
    return dir_name
    
def create_dir(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
        app.logger.info(f"Ensured upload directory exists: {dir_path}")
        return True
    except OSError as e:
        app.logger.error(f"Error creating directory {dir_path}: {e}")
        return False
        
@app.route('/')
def index():
    app.logger.info("Serving the index page.")
    return render_template("index.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        app.logger.info("Received a file upload POST request.")
        uploaded_files = request.files.getlist("file[]")
        filenames = []
        
        date_folder_name = get_dir_name()
        target_directory_path = os.path.join(app.config['UPLOAD_FOLDER'], date_folder_name)
        app.logger.debug(f"Target upload directory path: {target_directory_path}")
        
        if not create_dir(target_directory_path):
            return "Server error: Could not create upload directory.", 500
        
        filenames = []
        for file in uploaded_files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_save_path = os.path.join(target_directory_path, filename)
                file.save(file_save_path)
                
                relative_path_for_link = os.path.join(date_folder_name, filename)
                filenames.append(relative_path_for_link)
                
                app.logger.info(f"File saved to: {file_save_path}")
                
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
   


