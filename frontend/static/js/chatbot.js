const userId = localStorage.getItem("x-user-id");
if (!userId) {
  window.location.href = "/";
}

const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("question");
const imageUriInput = document.getElementById("imageUri");
const opsInput = document.getElementById("ops");
const logoutBtn = document.getElementById("logoutBtn");

function appendMessage(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

chatForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    return;
  }

  const payload = { question, top_k: 3 };
  const imageUri = imageUriInput.value.trim();
  if (imageUri) {
    payload.image_uri = imageUri;
  }

  const ops = opsInput.value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  if (ops.length > 0) {
    payload.ops = ops;
  }

  appendMessage("user", question);
  questionInput.value = "";

  try {
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
  } catch {
    appendMessage("bot", "서버 연결 실패");
  }
});

logoutBtn?.addEventListener("click", () => {
  localStorage.removeItem("x-user-id");
  window.location.href = "/";
});
