// main.js - Quantum Children shared functionality

// Neural Network Background Animation
class NeuralNetwork {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        this.resize();

        window.addEventListener('resize', () => this.resize());
        this.init();
        this.animate();
    }

    resize() {
        this.width = this.canvas.width = window.innerWidth;
        this.height = this.canvas.height = window.innerHeight;
    }

    init() {
        for (let i = 0; i < 80; i++) {
            this.particles.push({
                x: Math.random() * this.width,
                y: Math.random() * this.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5
            });
        }
    }

    animate() {
        this.ctx.clearRect(0, 0, this.width, this.height);

        this.particles.forEach(p => {
            // Update position
            p.x += p.vx;
            p.y += p.vy;

            // Bounce off walls
            if (p.x < 0 || p.x > this.width) p.vx *= -1;
            if (p.y < 0 || p.y > this.height) p.vy *= -1;

            // Draw particle
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
            this.ctx.fillStyle = '#ff0000';
            this.ctx.fill();

            // Draw connections
            this.particles.forEach(p2 => {
                const dx = p.x - p2.x;
                const dy = p.y - p2.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 120) {
                    this.ctx.strokeStyle = `rgba(255, 0, 0, ${0.1 * (1 - dist / 120)})`;
                    this.ctx.beginPath();
                    this.ctx.moveTo(p.x, p.y);
                    this.ctx.lineTo(p2.x, p2.y);
                    this.ctx.stroke();
                }
            });
        });

        requestAnimationFrame(() => this.animate());
    }
}

// Audio Controller
class AudioController {
    constructor() {
        this.audio = document.getElementById('bg-music');
        this.btn = document.getElementById('btn-audio');
        this.isPlaying = false;

        if (this.btn) {
            this.btn.addEventListener('click', () => this.toggle());
        }
    }

    play() {
        if (this.audio) {
            this.audio.volume = 0.4;
            this.audio.play().catch(() => {});
            this.isPlaying = true;
            if (this.btn) this.btn.textContent = 'MUTE AUDIO';
        }
    }

    toggle() {
        if (!this.audio) return;

        if (this.isPlaying) {
            this.audio.pause();
            this.isPlaying = false;
            if (this.btn) this.btn.textContent = 'PLAY AUDIO';
        } else {
            this.play();
        }
    }
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    // Start neural network animation
    new NeuralNetwork('neural-canvas');

    // Initialize audio controller
    window.audioController = new AudioController();

    // Handle intro overlay click (if exists)
    const overlay = document.getElementById('intro-overlay');
    if (overlay) {
        overlay.addEventListener('click', () => {
            overlay.style.display = 'none';
            window.audioController.play();
        });
    }
});

// Fetch network stats from collection server
async function fetchNetworkStats() {
    try {
        const response = await fetch('http://203.161.61.61:8888/stats');
        return await response.json();
    } catch (error) {
        console.log('Stats server offline or unreachable');
        return null;
    }
}
