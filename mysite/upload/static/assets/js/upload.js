function addInfo(data) {
  const inf = data.info;
  $("#maso").text(inf.maso);
  $("#hoten").text(inf.hoten);
  $("#ngaysinh").text(inf.ngaysinh);
  $("#nguyenquan").text(inf.nguyenquan);
  $("#diachi").text(inf.diachi);
}
function postData(input) {
  let formData = new FormData();
  formData.append("image", input.files[0]);
  waitLoader(true);
  $.ajax({
    url: "demo-cmt/",
    type: "post",
    data: formData,
    contentType: false,
    processData: false,
    success: function (data) {
      addInfo(data);
      waitLoader(false);
    },
  });
}

function waitLoader(f) {
  if (f) {
    $(".loader").show();
    $(".info-wrap").hide();
  } else {
    $(".loader").hide();
    $(".info-wrap").show();
  }
}

function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      $("#imageResult").attr("src", e.target.result);
    };
    reader.readAsDataURL(input.files[0]);
  }
}

$(function () {
  waitLoader(false);
  $("#upload").on("change", function () {
    readURL(input);
    postData(input);
  });
});

/*  ==========================================
    SHOW UPLOADED IMAGE NAME
* ========================================== */
var input = document.getElementById("upload");
var infoArea = document.getElementById("upload-label");

input.addEventListener("change", showFileName);
function showFileName(event) {
  var input = event.srcElement;
  var fileName = input.files[0].name;
  infoArea.textContent = "File name: " + fileName;
}
