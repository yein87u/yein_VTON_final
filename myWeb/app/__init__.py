from flask import Flask, redirect
from app.route import index, get_warping_result

def create_app():
    app = Flask(__name__)
    # 註冊根路徑，重定向到 /index
    app.add_url_rule('/', 'root', lambda: redirect('/index'))

    # 主頁面
    app.add_url_rule('/index', 'index', index)

    # 到此頁面會將圖片導入服裝變形模組當中
    app.add_url_rule('/index/get_warping_result', 
                     'get_warping_result', get_warping_result, 
                     methods=['POST'])  # 預測請求
    return app