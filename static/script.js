let slideIndex = 0;
showSlides(slideIndex);

function moveCarousel(n) {
    showSlides((slideIndex += n));
}

function showSlides(n) {
    let slides = document.getElementsByClassName("carousel-slide");
    if (n >= slides.length) {
        slideIndex = 0;
    }
    if (n < 0) {
        slideIndex = slides.length - 1;
    }
    for (let i = 0; i < slides.length; i++) {
        slides[i].style.transform = "translateX(-" + slideIndex * 100 + "%)";
    }
}
document.getElementById("prevBtn").addEventListener("click", function () {
    moveCarousel(-1);
});

document.getElementById("nextBtn").addEventListener("click", function () {
    moveCarousel(1);
});

// Auto-slide every 2 seconds
setInterval(function () {
    moveCarousel(1);
}, 2000);
