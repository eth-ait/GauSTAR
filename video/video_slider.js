const videos = ["video/res_241213T8_rot_min.mp4", "video/res_241111T2_rot_min.mp4", "video/res_241213T5_rot_min.mp4",  "video/res_240906T3_rot_min.mp4", "video/res_220920T1_rot_min.mp4", "video/res_241028T4_rot_min.mp4", ];
let currentIndex = 0;
let userInteracted = false;

const videoPreview = document.getElementById('videoPreview');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');

// Dynamically add video elements
function initializeVideos() {
    videos.forEach((videoSrc, index) => {
        const videoItem = document.createElement('div');
        videoItem.className = 'video-item';
        if (index === currentIndex) videoItem.classList.add('active');

        const videoElement = document.createElement('video');
        videoElement.muted = true;
        videoElement.autoplay = true;
        videoElement.loop = true;

        const source = document.createElement('source');
        source.src = videoSrc;
        source.type = 'video/mp4';

        videoElement.appendChild(source);
        videoItem.appendChild(videoElement);
        videoPreview.appendChild(videoItem);
    });
}

function updatePreview() {
    const items = document.querySelectorAll('.video-item');
    items.forEach((item, index) => {
        const video = item.querySelector('video');
        if (index === currentIndex) {
            item.classList.add('active');
            video.controls = true;
            video.muted = !userInteracted;
            if (userInteracted) video.play();
        } else {
            item.classList.remove('active');
            video.controls = false;
            video.muted = true;
            video.pause();
        }
    });

    const offset = -(currentIndex * 410) + (1100 / 2 - 450);
    videoPreview.style.transform = `translateX(${offset}px)`;
}

function handleUserInteraction() {
    if (!userInteracted) {
        userInteracted = true;
        const activeVideo = document.querySelector('.video-item.active video');
        activeVideo.muted = false;
        activeVideo.play().catch((err) => console.warn("Play failed:", err));
    }
}

prevBtn.addEventListener('click', () => {
    handleUserInteraction();
    currentIndex = (currentIndex - 1 + videos.length) % videos.length;
    updatePreview();
});

nextBtn.addEventListener('click', () => {
    handleUserInteraction();
    currentIndex = (currentIndex + 1) % videos.length;
    updatePreview();
});

// Initialize videos and set the initial state
window.addEventListener('DOMContentLoaded', () => {
    initializeVideos();
    updatePreview();
});