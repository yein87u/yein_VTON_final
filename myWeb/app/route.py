# 渲染html
from flask import request, render_template
# 引入模型
from app.models import WarpModel

# 渲染html
def index():
    return render_template('index.html') 

def get_warping_result():
    print("get_warping_result!")
     # 處理上傳的圖片
    origin_person_image = request.files['origin_person_image']
    want_tryon_cloth = request.files['want_tryon_cloth']
    print("origin_person_image: ", origin_person_image)
    
    
    # 假設有一個函數來保存上傳的圖片
    # image_path = save_image(image)
    
    # 在 Ubuntu 虛擬機器上運行模型
    result = WarpModel(origin_person_image)
    
    # 返回結果到渲染頁面
    return render_template('result.html', result=result)