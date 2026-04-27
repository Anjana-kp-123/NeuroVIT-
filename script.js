async function uploadMRI() {

    const fileInput = document.getElementById("imageUpload");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please upload an MRI image");
        return;
    }

    // preview image
    document.getElementById("preview").src = URL.createObjectURL(file);

    const formData = new FormData();
    formData.append("image", file);

    const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    // Show prediction
    document.getElementById("result").innerHTML =
        "Prediction: <b>" + data.prediction +
        "</b><br>Confidence: " + data.confidence;

    //  Segmentation Output
    document.getElementById("maskImage").src =
        "data:image/png;base64," + data.segmentation_mask;

    document.getElementById("overlayImage").src =
        "data:image/png;base64," + data.overlay_image;
}