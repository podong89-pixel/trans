import os

import openai
import streamlit as st
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

DEFAULT_MODEL = "gpt-6-astra"
MAX_CHARS = 5000
# 버튼 라벨 → (언어 코드, 프롬프트에 전달할 언어명)
# 국기 이모지는 Windows에서 "US" 같은 글자로 깨져 보이므로 언어 코드를 사용
LANGUAGES = {
    "영어": ("EN", "English"),
    "일본어": ("JA", "Japanese"),
    "베트남어": ("VI", "Vietnamese"),
}
SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text into {target_language}. "
    "Output only the translated text without explanations, notes, or quotation marks. "
    "Preserve line breaks and formatting."
)

CSS = """
<style>
.block-container { max-width: 800px; padding-top: 2.5rem; }
h1 { word-break: keep-all; }
@media (max-width: 480px) {
    h1 { font-size: 1.9rem !important; }
}
.lang-code {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 0.1rem 0.4rem;
    border-radius: 0.3rem;
    background: rgba(91, 140, 255, 0.18);
    margin-right: 0.35rem;
    vertical-align: middle;
}
.app-subtitle { opacity: 0.75; margin-top: -0.6rem; margin-bottom: 1.2rem; }
.char-count { text-align: right; font-size: 0.85rem; opacity: 0.6; margin-top: -0.6rem; }
.char-count.over { color: #e5484d; opacity: 1; font-weight: 600; }
.result-label { font-weight: 600; margin: 1.4rem 0 0.4rem; }
.result-label span { opacity: 0.6; font-weight: 400; }
.result-empty {
    border: 1px dashed rgba(128, 128, 128, 0.45);
    border-radius: 0.6rem;
    padding: 1.4rem;
    text-align: center;
    opacity: 0.7;
    margin-top: 1.4rem;
}
</style>
"""


def load_config() -> tuple[str | None, str]:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip() or None
    model = (os.getenv("OPENAI_MODEL") or "").strip() or DEFAULT_MODEL
    return api_key, model


@st.cache_resource
def get_client(api_key: str) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key)


def translate(client: openai.OpenAI, model: str, text: str, target_language: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(target_language=target_language)},
            {"role": "user", "content": text},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def validate_input(text: str) -> str | None:
    if not text.strip():
        return "번역할 텍스트를 입력해 주세요."
    if len(text) > MAX_CHARS:
        return f"최대 {MAX_CHARS:,}자까지 입력할 수 있습니다."
    return None


def error_message(exc: Exception, model: str) -> str:
    if isinstance(exc, openai.AuthenticationError):
        return "API 키가 올바르지 않습니다."
    if isinstance(exc, (openai.NotFoundError, openai.PermissionDeniedError)):
        return f"모델({model})을 사용할 수 없습니다. 모델 이름과 계정 권한을 확인해 주세요."
    if isinstance(exc, openai.RateLimitError):
        return "요청이 많습니다. 잠시 후 다시 시도해 주세요."
    return f"번역 중 오류가 발생했습니다: {exc}"


def clear_all() -> None:
    st.session_state["text"] = ""
    st.session_state.pop("result", None)
    st.session_state.pop("result_label", None)


def render_result() -> None:
    result = st.session_state.get("result")
    if result is None:
        st.markdown(
            '<div class="result-empty">언어 버튼을 누르면 이곳에 번역 결과가 표시됩니다.</div>',
            unsafe_allow_html=True,
        )
        return
    label = st.session_state["result_label"]
    code, english_name = LANGUAGES[label]
    st.markdown(
        f'<div class="result-label"><span class="lang-code">{code}</span>번역 결과 '
        f'<span>· {english_name}</span></div>',
        unsafe_allow_html=True,
    )
    st.code(result, language=None, wrap_lines=True)


api_key, model = load_config()

st.set_page_config(page_title="다국어 번역기", page_icon="🌐", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ 설정 정보")
    st.markdown(f"**모델:** `{model}`")
    st.caption("모델은 .env 파일의 OPENAI_MODEL 값으로 변경할 수 있습니다.")

st.title("🌐 다국어 번역기")
st.markdown(
    '<p class="app-subtitle">번역할 글을 입력하고 언어 버튼을 눌러주세요.</p>',
    unsafe_allow_html=True,
)

if not api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
    st.stop()

text = st.text_area(
    "번역할 텍스트",
    key="text",
    placeholder="번역할 텍스트를 입력하세요...",
    height=200,
    label_visibility="collapsed",
)
over = " over" if len(text) > MAX_CHARS else ""
st.markdown(
    f'<div class="char-count{over}">{len(text):,} / {MAX_CHARS:,}</div>',
    unsafe_allow_html=True,
)

clicked = None
cols = st.columns(3)
for col, (label, (code, _)) in zip(cols, LANGUAGES.items()):
    is_last = st.session_state.get("result_label") == label
    if col.button(
        f"{label} · {code}",
        key=f"btn_{label}",
        type="primary" if is_last else "secondary",
        use_container_width=True,
    ):
        clicked = label

st.button("🧹 입력 지우기", on_click=clear_all, type="tertiary")

if "warning" in st.session_state:
    st.warning(st.session_state.pop("warning"))

if clicked:
    problem = validate_input(text)
    if problem:
        # 경고와 함께 이전 번역 결과가 남아 현재 입력의 결과처럼 보이지 않도록 제거하고,
        # 버튼 강조 상태도 함께 갱신되도록 다시 그린 뒤 경고를 표시
        st.session_state.pop("result", None)
        st.session_state.pop("result_label", None)
        st.session_state["warning"] = problem
        st.rerun()
    else:
        try:
            with st.spinner("번역 중..."):
                result = translate(get_client(api_key), model, text, LANGUAGES[clicked][1])
            st.session_state["result"] = result
            st.session_state["result_label"] = clicked
            st.rerun()  # 마지막 사용 언어 버튼을 primary로 다시 그리기 위해
        except Exception as exc:
            st.error(error_message(exc, model))

render_result()
