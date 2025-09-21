// 這邊獲取的是HTML表單內的id屬性
const personImage = document.getElementById("origin_person_image");
const clothImage = document.getElementById("want_tryon_cloth");
const personPreview = document.getElementById("show_origin_person_image");
const clothPreview = document.getElementById("show_want_tryon_cloth");
const result = document.getElementById("result");

// 預覽上傳的圖片
// 圖片檔案名稱 file.name
personImage.addEventListener("change", function() {
    const file = this.files[0];
    if (file) {
        personPreview.src = URL.createObjectURL(file);
    }
});

clothImage.addEventListener("change", function() {
    const file = this.files[0];
    if (file) {
        clothPreview.src = URL.createObjectURL(file);
    }
});

// 當服裝變形按鈕按下
function ClothWarpingGenerate() {
    const formData = new FormData();
    const person = personImage.files[0];
    const cloth = clothImage.files[0];
    if (!person || !cloth) {
        alert("請選擇兩張圖片！");
        return;
    }

    // 這裡指的是HTML內表單中的name屬性
    formData.append("origin_person_image", person);
    formData.append("want_tryon_cloth", cloth);

    fetch("/index/get_warping_result", {method: "POST", body: formData})
    .then(res => res.blob()) 
    .then(blob => {
        result.src = URL.createObjectURL(blob);  // 顯示生成圖
    })
    .catch(err => alert("發生錯誤：" + err));
}