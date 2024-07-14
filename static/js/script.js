document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('imageUpload');
    const uploadedImage = document.getElementById('uploadedImage');

    fileInput.addEventListener('change', function(event) {
        const file = event.target.files[0];

        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                uploadedImage.src = e.target.result;
                uploadedImage.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });
});