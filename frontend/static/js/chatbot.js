const userId = localStorage.getItem("x-user-id");
if (!userId) {
  window.location.href = "/";
}

const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("question");
const imageUriInput = document.getElementById("imageUri");
const imageFileInput = document.getElementById("imageFile");
const opsInput = document.getElementById("ops");
const logoutBtn = document.getElementById("logoutBtn");

function appendMessage(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function appendImageMessage(role, src, caption = "") {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;

  const image = document.createElement("img");
  image.src = src;
  image.alt = "uploaded-image";
  image.className = "chat-image";
  wrapper.appendChild(image);

  if (caption) {
    const text = document.createElement("div");
    text.className = "chat-image-caption";
    text.textContent = caption;
    wrapper.appendChild(text);
  }

  chatLog.appendChild(wrapper);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function toPreviewSrc(imageUri) {
  if (!imageUri) {
    return "";
  }
  if (imageUri.startsWith("http://") || imageUri.startsWith("https://") || imageUri.startsWith("/")) {
    return imageUri;
  }
  if (imageUri.startsWith("data/imagery/")) {
    return imageUri.replace("data/imagery", "/imagery");
  }
  return imageUri;
}

async function uploadSelectedImage() {
  const file = imageFileInput?.files?.[0];
  if (!file) {
    return { imageUri: null, previewUrl: null, fileName: null };
  }

  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/upload-image", {
    method: "POST",
    headers: { "x-user-id": userId },
    body: formData,
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || "이미지 업로드 실패");
  }

  const data = await response.json();
  return {
    imageUri: data.image_uri || null,
    previewUrl: data.preview_url || toPreviewSrc(data.image_uri || ""),
    fileName: file.name,
  };
}

chatForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  const hasImageUri = !!imageUriInput.value.trim();
  const hasImageFile = !!imageFileInput?.files?.length;
  if (!question && !hasImageUri && !hasImageFile) {
    appendMessage("bot", "질문 또는 이미지를 입력해 주세요.");
    return;
  }

  const payload = { question, top_k: 3 };

  const ops = opsInput.value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  if (ops.length > 0) {
    payload.ops = ops;
  }

  try {
    let imageUri = imageUriInput.value.trim();
    let previewSrc = toPreviewSrc(imageUri);

    // Always upload when a file is selected, so image_uri updates automatically.
    if (hasImageFile) {
      const uploaded = await uploadSelectedImage();
      imageUri = uploaded.imageUri || imageUri;
      previewSrc = uploaded.previewUrl || toPreviewSrc(imageUri);
      if (imageUri) {
        imageUriInput.value = imageUri;
      }
      if (imageFileInput) {
        imageFileInput.value = "";
      }
      appendImageMessage("user", previewSrc, uploaded.fileName || "uploaded image");
    } else if (imageUri) {
      appendImageMessage("user", previewSrc, "image uri");
    }

    if (imageUri) {
      payload.image_uri = imageUri;
    }

    appendMessage("user", question || "(질문 없음, 이미지 분석 요청)");
    questionInput.value = "";

    const response = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-user-id": userId,
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      appendMessage("bot", `요청 실패: ${data.detail || response.status}`);
      return;
    }

    const data = await response.json();
    const citationCount = data.citations?.length || 0;
    const analysisSummary = data.analysis?.invoked
      ? `\n[analysis] ${JSON.stringify(data.analysis.ops || [])}`
      : "";
    appendMessage("bot", `${data.answer}\n[citations] ${citationCount}${analysisSummary}`);
  } catch (error) {
    appendMessage("bot", `서버 연결 실패 또는 업로드 실패: ${error.message || "unknown"}`);
  }
});

logoutBtn?.addEventListener("click", () => {
  localStorage.removeItem("x-user-id");
  window.location.href = "/";
});
