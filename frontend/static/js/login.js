const loginBtn = document.getElementById("loginBtn");
const userInput = document.getElementById("userId");
const errorBox = document.getElementById("loginError");

async function tryLogin() {
  const userId = userInput.value.trim();
  if (!userId) {
    errorBox.textContent = "user id를 입력하세요.";
    return;
  }

  errorBox.textContent = "";
  try {
    const response = await fetch("/auth/verify", {
      method: "GET",
      headers: { "x-user-id": userId },
    });

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      errorBox.textContent = data.detail || "로그인 실패";
      return;
    }

    localStorage.setItem("x-user-id", userId);
    window.location.href = "/chatbot";
  } catch {
    errorBox.textContent = "서버 연결 실패";
  }
}

loginBtn?.addEventListener("click", tryLogin);
userInput?.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    tryLogin();
  }
});
