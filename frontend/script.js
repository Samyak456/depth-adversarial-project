async function uploadImage() {
    const fileInput = document.getElementById("imageInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    displayResults(data);
}

function displayResults(data) {
    const div = document.getElementById("results");

    div.innerHTML = `
        <h3>Status: ${data.status}</h3>
        <h3>${data.detection.warning}</h3>

        <h4>Metrics:</h4>
        <p>Noise: ${data.metrics.noise_error}</p>
        <p>Patch: ${data.metrics.patch_error}</p>
        <p>Stripes: ${data.metrics.stripes_error}</p>

        <h4>Visualization:</h4>
        <img src="${data.visualization}">
    `;
}