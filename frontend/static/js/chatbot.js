const userId = localStorage.getItem("x-user-id");
if (!userId) {
  window.location.href = "/";
}

const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("question");
const imageFileInput = document.getElementById("imageFile");
const logoutBtn = document.getElementById("logoutBtn");
const llmStatus = document.getElementById("llmStatus");
let typingBubble = null;
let typingRawText = "";

function appendTextMessage(role, text) {
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
  image.src = resolveRenderableImageSrc(src);
  image.alt = "uploaded-image";
  image.className = "chat-image";
  attachImageFallback(image, src);
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

function appendStructuredResponse(data) {
  const wrapper = document.createElement("div");
  wrapper.className = "msg bot";

  const title = document.createElement("div");
  title.className = "resp-title";
  title.textContent = "Response Summary";
  wrapper.appendChild(title);

  const answer = document.createElement("div");
  answer.className = "resp-section";
  const answerLabel = document.createElement("strong");
  answerLabel.textContent = "Answer";
  answer.appendChild(answerLabel);
  answer.appendChild(document.createElement("br"));
  appendRichContent(answer, data.answer || "");
  wrapper.appendChild(answer);

  const tools = data.trace?.tools || [];
  const ragRoute = tools.find((tool) => tool.startsWith("route.rag:")) || "route.rag:false";
  const mcpRoute = tools.find((tool) => tool.startsWith("route.mcp:")) || "route.mcp:false";

  const trace = document.createElement("div");
  trace.className = "resp-section";
  trace.innerHTML = `<strong>Pipeline</strong><br>${escapeHtml(tools.join(", "))}<br>latency: ${data.trace?.latency_ms || 0} ms`;
  wrapper.appendChild(trace);

  const rag = document.createElement("div");
  rag.className = "resp-section";
  rag.innerHTML = `<strong>RAG</strong><br>decision: ${escapeHtml(ragRoute)}<br>hits: ${data.citations?.length || 0}`;
  if ((data.citations?.length || 0) > 0) {
    const ul = document.createElement("ul");
    ul.className = "resp-list";
    data.citations.forEach((item) => {
      const li = document.createElement("li");
      const lineInfo =
        item.line_start && item.line_end
          ? `lines ${item.line_start}-${item.line_end}`
          : item.chunk_id;
      li.textContent = `${item.doc_id} | ${lineInfo} | score=${item.score}`;
      ul.appendChild(li);
    });
    rag.appendChild(ul);
  }
  wrapper.appendChild(rag);

  const mcp = document.createElement("div");
  mcp.className = "resp-section";
  const opNames = (data.analysis?.ops || []).map((op) => op.name).join(", ") || "(none)";
  mcp.innerHTML = `<strong>MCP</strong><br>decision: ${escapeHtml(mcpRoute)}<br>ops: ${escapeHtml(opNames)}`;
  if (data.analysis?.error) {
    const err = document.createElement("div");
    err.className = "resp-error";
    err.textContent = `error: ${data.analysis.error}`;
    mcp.appendChild(err);
  }
  if (data.analysis?.ops?.length) {
    const ul = document.createElement("ul");
    ul.className = "resp-list";
    data.analysis.ops.forEach((op) => {
      const li = document.createElement("li");
      li.textContent = `${op.name}: ${op.summary} | stats=${JSON.stringify(op.stats || {})}`;
      if (op.artifact_uri) {
        const artifact = document.createElement("img");
        artifact.src = resolveRenderableImageSrc(op.artifact_uri);
        artifact.alt = `${op.name}-artifact`;
        artifact.className = "artifact-image";
        attachImageFallback(artifact, op.artifact_uri);
        li.appendChild(artifact);
      }
      ul.appendChild(li);
    });
    mcp.appendChild(ul);
  }
  wrapper.appendChild(mcp);

  chatLog.appendChild(wrapper);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function startTypingBubble() {
  if (typingBubble) {
    return;
  }
  typingRawText = "";
  typingBubble = document.createElement("div");
  typingBubble.className = "msg bot typing";
  typingBubble.textContent = "";
  chatLog.appendChild(typingBubble);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function appendTypingChunk(text) {
  if (!typingBubble) {
    startTypingBubble();
  }
  typingRawText += text;
  typingBubble.textContent = sanitizeDisplayText(typingRawText);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function endTypingBubble() {
  if (!typingBubble) {
    return;
  }
  typingBubble.remove();
  typingBubble = null;
  typingRawText = "";
}

function appendStatus(event) {
  if (event.stage === "route") {
    setProgressStatus("도구 판단 중");
    return;
  }
  if (event.stage === "llm") {
    setProgressStatus("LLM 답변 생성 중");
    return;
  }
  if (event.stage === "rag") {
    setProgressStatus(event.used ? `RAG 검색 완료 (hits ${event.hits})` : "RAG 스킵");
    return;
  }
  if (event.stage === "mcp") {
    const opSummary = (event.ops || []).join(", ") || "none";
    setProgressStatus(event.invoked ? `MCP 분석 (${opSummary})` : "MCP 스킵");
    return;
  }
  setProgressStatus("처리 중");
}

function setProgressStatus(text) {
  if (!llmStatus) {
    return;
  }
  llmStatus.textContent = text;
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function stripMarkdownImageLinks(text) {
  return text.replaceAll(/!\[[^\]]*]\([^)]+\)/g, "").trim();
}

function stripUrls(text) {
  return text.replaceAll(/https?:\/\/[^\s)]+/g, "").trim();
}

function sanitizeDisplayText(text) {
  return stripUrls(stripMarkdownImageLinks(text));
}

function normalizeImageUrl(url) {
  if (url.startsWith("data/imagery/")) {
    return `/imagery/${url.slice("data/imagery/".length)}`;
  }
  if (url.startsWith("imagery/")) {
    return `/${url}`;
  }
  return url;
}

function isImageUrl(url) {
  return /\.(png|jpe?g|gif|webp|bmp|tiff?)(\?.*)?$/i.test(url) || url.startsWith("/imagery/");
}

function resolveRenderableImageSrc(src) {
  if (!src) {
    return "";
  }
  return normalizeImageUrl(src.trim());
}

function attachImageFallback(image, rawSrc) {
  image.addEventListener("error", () => {
    const fallback = document.createElement("div");
    fallback.className = "chat-image-caption";
    fallback.textContent = `이미지 로드 실패: ${rawSrc}`;
    image.replaceWith(fallback);
  });
}

function appendRichContent(container, text) {
  const regex = /!\[([^\]]*)]\(([^)\s]+)\)|(https?:\/\/[^\s)]+|\/[^\s)]+|data\/imagery\/[^\s)]+)/g;
  let cursor = 0;
  let matched = false;
  let match;

  while ((match = regex.exec(text)) !== null) {
    const [raw, alt, markdownUrl, plainUrl] = match;
    const start = match.index;
    if (start > cursor) {
      container.appendChild(document.createTextNode(text.slice(cursor, start)));
    }

    const candidateUrl = resolveRenderableImageSrc(markdownUrl || plainUrl || "");
    if (candidateUrl && isImageUrl(candidateUrl)) {
      const image = document.createElement("img");
      image.src = candidateUrl;
      image.alt = alt || "response-image";
      image.className = "chat-image";
      attachImageFallback(image, markdownUrl || plainUrl || "");
      container.appendChild(document.createElement("br"));
      container.appendChild(image);
      container.appendChild(document.createElement("br"));
      matched = true;
    } else {
      container.appendChild(document.createTextNode(raw));
    }
    cursor = start + raw.length;
  }

  if (cursor < text.length) {
    container.appendChild(document.createTextNode(text.slice(cursor)));
  }
  if (!matched && !text) {
    container.appendChild(document.createTextNode(""));
  }
}

async function uploadSelectedImage() {
  const file = imageFileInput?.files?.[0];
  if (!file) {
    return null;
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

  return response.json();
}

async function readNdjsonStream(response) {
  if (!response.body) {
    throw new Error("stream body is missing");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalData = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.trim()) {
        continue;
      }
      const event = JSON.parse(line);
      if (event.type === "status") {
        appendStatus(event);
      } else if (event.type === "answer_start") {
        startTypingBubble();
      } else if (event.type === "answer_chunk") {
        appendTypingChunk(event.text || "");
      } else if (event.type === "final") {
        finalData = event.data;
      }
    }
  }

  return finalData;
}

chatForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  const hasImageFile = !!imageFileInput?.files?.length;

  if (!question && !hasImageFile) {
    appendTextMessage("bot", "질문 또는 이미지를 입력해 주세요.");
    return;
  }

  try {
    setProgressStatus("요청 전송 중");
    let imageUri = "";

    if (hasImageFile) {
      const uploaded = await uploadSelectedImage();
      imageUri = uploaded.image_uri || "";
      const previewUrl = uploaded.preview_url || "";
      if (previewUrl) {
        appendImageMessage("user", previewUrl, imageFileInput.files[0]?.name || "uploaded image");
      }
      imageFileInput.value = "";
    }

    appendTextMessage("user", question || "(질문 없음, 이미지 분석 요청)");
    questionInput.value = "";

    const payload = { question, top_k: 3 };
    if (imageUri) {
      payload.image_uri = imageUri;
    }

    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-user-id": userId,
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      appendTextMessage("bot", `요청 실패: ${data.detail || response.status}`);
      setProgressStatus("요청 실패");
      return;
    }

    const finalData = await readNdjsonStream(response);
    endTypingBubble();
    if (finalData) {
      appendStructuredResponse(finalData);
      setProgressStatus("완료");
    } else {
      appendTextMessage("bot", "스트리밍 응답이 비어 있습니다.");
      setProgressStatus("응답 없음");
    }
  } catch (error) {
    appendTextMessage("bot", `요청 실패: ${error.message || "unknown error"}`);
    setProgressStatus("오류");
  }
});

logoutBtn?.addEventListener("click", () => {
  localStorage.removeItem("x-user-id");
  window.location.href = "/";
});
