function buildStarGradients(count, width, height, minSize, maxSize, alphaMin, alphaMax) {
  const colors = ["255,255,255", "205,225,255", "170,205,255"];
  const gradients = [];
  for (let i = 0; i < count; i += 1) {
    const x = Math.floor(Math.random() * width);
    const y = Math.floor(Math.random() * height);
    const size = (Math.random() * (maxSize - minSize) + minSize).toFixed(2);
    const alpha = (Math.random() * (alphaMax - alphaMin) + alphaMin).toFixed(2);
    const color = colors[Math.floor(Math.random() * colors.length)];
    gradients.push(`radial-gradient(${size}px ${size}px at ${x}px ${y}px, rgba(${color}, ${alpha}), transparent)`);
  }
  return gradients.join(",");
}

function renderRandomSpace() {
  const layer1 = document.querySelector(".stars");
  const layer2 = document.querySelector(".stars-2");
  if (!layer1 || !layer2) {
    return;
  }

  const w = Math.max(window.innerWidth, 1200);
  const h = Math.max(window.innerHeight, 900);

  layer1.style.backgroundImage = buildStarGradients(120, w, h, 0.9, 2.4, 0.22, 0.95);
  layer2.style.backgroundImage = buildStarGradients(80, w, h, 0.6, 1.7, 0.16, 0.75);
}

renderRandomSpace();
window.addEventListener("resize", () => {
  window.clearTimeout(window.__spaceResizeTimer);
  window.__spaceResizeTimer = window.setTimeout(renderRandomSpace, 140);
});
