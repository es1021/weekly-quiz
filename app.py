import io
import json
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types

# ---------- 기본 설정 ----------
st.set_page_config(page_title="주간 퀴즈 학습", page_icon="📚", layout="wide")


# ---------- 비밀번호 잠금 ----------
def check_password():
    try:
        pw = st.secrets.get("APP_PASSWORD")
    except Exception:
        pw = None
    if not pw:
        return  # 비밀번호 미설정 시(로컬 테스트) 통과
    if st.session_state.get("authed"):
        return
    st.title("🔒 주간 퀴즈 학습")
    entered = st.text_input("비밀번호", type="password")
    if entered == pw:
        st.session_state["authed"] = True
        st.rerun()
    elif entered:
        st.error("비밀번호가 틀렸어요.")
    st.stop()


check_password()

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
DB_FILE = DATA_DIR / "db.json"
PDF_DIR.mkdir(parents=True, exist_ok=True)

LANGS = ["java", "python", "c", "cpp", "sql", "text"]


# ---------- 저장소 ----------
def load_db():
    db = {"subjects": {}}
    if DB_FILE.exists():
        db = json.loads(DB_FILE.read_text(encoding="utf-8"))
    if "weeks" in db:  # 과목 구분이 없던 구버전 데이터 → '기본' 과목으로 이동
        db.setdefault("subjects", {})
        db["subjects"].setdefault("기본", {"weeks": {}})["weeks"].update(db.pop("weeks"))
    db.setdefault("subjects", {})
    return db


def save_db(db):
    DB_FILE.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")


def new_week():
    return {
        "pdf": None,
        "important": "",
        "summary": "",
        "snippets": [],
        "memo": "",
        "quiz": [],
        "history": [],
        "wrong": [],
    }


def pdf_path(subject, week):
    safe = re.sub(r"[^\w가-힣-]", "_", f"{subject}_{week}")
    return PDF_DIR / f"{safe}.pdf"


# ---------- Gemini ----------
def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
    return key or st.session_state.get("api_key_input")


def ask_gemini(prompt, pdf_bytes=None, model="gemini-3.8-flash", as_json=True):
    key = get_api_key()
    if not key:
        st.error("사이드바에 Gemini API 키를 입력하세요.")
        st.stop()
    client = genai.Client(api_key=key)
    contents = []
    if pdf_bytes:
        contents.append(types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"))
    contents.append(prompt)
    config = types.GenerateContentConfig(response_mime_type="application/json") if as_json else None
    models = [model]
    fallback = (st.session_state.get("fallback_model") or "").strip()
    if fallback and fallback != model:
        models.append(fallback)
    retryable = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500")
    last_err = None
    resp = None
    for m in models:
        for attempt in range(3):  # 일시적 과부하(503 등)는 자동 재시도
            try:
                resp = client.models.generate_content(model=m, contents=contents, config=config)
                break
            except Exception as e:
                last_err = e
                if not any(c in str(e) for c in retryable):
                    raise
                time.sleep(3 * (attempt + 1))
        if resp is not None:
            break
    if resp is None:
        raise last_err
    text = resp.text.strip()
    if not as_json:
        return text
    text = re.sub(r"^```(?:json)?|```$", "", text).strip()
    return json.loads(text)


def generate_summary(w, level, model):
    pdf_bytes = Path(w["pdf"]).read_bytes()
    extra = ""
    if w["important"].strip() or w["snippets"]:
        extra = "\n학생이 표시한 아래 '중요 부분'은 더 자세히 다뤄라.\n" + build_context(w, with_summary=False)
    prompt = f"""첨부된 강의 PDF를 한국어로 요약해라. 요약 분량: {level}.
마크다운 형식으로 아래 구성을 따른다.

## 📌 한 줄 요약
## 🔑 핵심 개념 (정의 포함)
## 📖 상세 정리 (PDF 흐름에 따라 섹션별로)
## ⚠️ 시험에 나올 만한 포인트
## 📚 용어 정리 (표)

코드가 있으면 코드 블록으로 보여주고 핵심 동작을 설명해라.{extra}"""
    return ask_gemini(prompt, pdf_bytes, model, as_json=False)


def build_context(w, with_summary=True):
    parts = []
    if with_summary and w.get("summary", "").strip():
        parts.append("[강의 요약]\n" + w["summary"])
    if w["important"].strip():
        parts.append("[중요 부분 메모]\n" + w["important"])
    for s in w["snippets"]:
        parts.append(f"[중요 코드: {s['title']} ({s['lang']})]\n{s['code']}")
    return "\n\n".join(parts) or "(없음)"


def generate_quiz(w, n_mc, n_short, n_code, use_pdf, model):
    prompt = f"""너는 대학 강의 복습용 퀴즈 출제자다.
첨부된 강의 PDF(있다면)와 아래 '중요 부분'을 분석해 퀴즈를 만들어라.
중요 부분에 있는 내용은 더 높은 비중으로 출제한다.

{build_context(w)}

출제 개수: 객관식 {n_mc}개, 단답/서술형 {n_short}개, 코드 작성/빈칸 {n_code}개 (코드 문제는 Java 기준, 중요 코드가 있으면 그 언어).
반드시 아래 JSON 배열만 출력한다. 한국어로 작성한다.
[
  {{"type": "mc", "question": "...", "choices": ["...","...","...","..."], "answer": "정답 보기 문자열 그대로", "explanation": "..."}},
  {{"type": "short", "question": "...", "answer": "모범 답안", "explanation": "..."}},
  {{"type": "code", "question": "...", "answer": "모범 코드", "explanation": "..."}}
]"""
    pdf_bytes = None
    if use_pdf and w["pdf"] and Path(w["pdf"]).exists():
        pdf_bytes = Path(w["pdf"]).read_bytes()
    return ask_gemini(prompt, pdf_bytes, model)


def grade(quiz, answers, model):
    results = [None] * len(quiz)
    to_ai = []
    for i, q in enumerate(quiz):
        if q["type"] == "mc":
            ok = answers[i] == q["answer"]
            results[i] = {"correct": ok, "feedback": q.get("explanation", "")}
        else:
            to_ai.append(i)
    if to_ai:
        items = [
            {
                "id": i,
                "question": quiz[i]["question"],
                "model_answer": quiz[i]["answer"],
                "student_answer": answers[i] or "(무응답)",
            }
            for i in to_ai
        ]
        prompt = f"""다음 학생 답안을 채점해라. 의미가 맞으면 표현이 달라도 정답으로 본다.
코드 문제는 로직이 맞으면 정답으로 보고, 틀린 부분이 있으면 짧게 알려준다.
JSON 배열만 출력: [{{"id": 0, "correct": true, "feedback": "한두 문장 피드백"}}]

{json.dumps(items, ensure_ascii=False)}"""
        for r in ask_gemini(prompt, model=model):
            results[r["id"]] = {"correct": bool(r["correct"]), "feedback": r["feedback"]}
    return results


# ---------- 상태 ----------
db = load_db()

with st.sidebar:
    st.title("📚 주간 퀴즈 학습")
    try:
        has_secret = bool(st.secrets.get("GEMINI_API_KEY"))
    except Exception:
        has_secret = False
    if not has_secret:
        st.text_input("Gemini API 키", type="password", key="api_key_input")
    model = st.text_input("모델", value="gemini-3.8-flash")
    st.text_input("예비 모델 (선택)", key="fallback_model", help="기본 모델이 계속 과부하일 때 대신 쓸 모델 이름")

    st.divider()
    subjects = list(db["subjects"].keys())
    if subjects:
        subject = st.selectbox("과목 선택", subjects)
    else:
        subject = None
        st.info("아래에서 과목을 추가하세요.")
    new_subject = st.text_input("새 과목 이름", placeholder="예: 알고리즘")
    if st.button("➕ 과목 추가", use_container_width=True):
        name = new_subject.strip()
        if name and name not in db["subjects"]:
            db["subjects"][name] = {"weeks": {}}
            save_db(db)
            st.rerun()
        else:
            st.warning("이름이 비었거나 이미 있어요.")

    week = None
    if subject:
        st.divider()
        weeks_dict = db["subjects"][subject]["weeks"]
        weeks = list(weeks_dict.keys())
        if weeks:
            week = st.selectbox("주차 선택", weeks, index=len(weeks) - 1, key=f"week_sel_{subject}")
        else:
            st.info("아래에서 주차를 추가하세요.")
        new_name = st.text_input(
            "새 주차 이름", value=f"{len(weeks) + 1}주차", key=f"new_week_{subject}_{len(weeks)}"
        )
        if st.button("➕ 주차 추가", use_container_width=True):
            if new_name and new_name not in weeks_dict:
                weeks_dict[new_name] = new_week()
                save_db(db)
                st.rerun()
            else:
                st.warning("이름이 비었거나 이미 있어요.")

    st.divider()
    st.caption("무료 호스팅은 재부팅 시 파일이 사라질 수 있어요. 백업을 받아두세요.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in DATA_DIR.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(DATA_DIR))
    st.download_button("💾 전체 백업(zip)", buf.getvalue(), "backup.zip", use_container_width=True)
    restore = st.file_uploader("백업 복원", type="zip", key="restore")
    if restore and st.button("복원 실행", use_container_width=True):
        with zipfile.ZipFile(restore) as z:
            z.extractall(DATA_DIR)
        st.rerun()

if not subject:
    st.title("과목을 먼저 추가하세요 👈")
    st.stop()
if not week:
    st.title(subject)
    st.info("사이드바에서 주차를 추가하세요 👈")
    st.stop()

w = db["subjects"][subject]["weeks"][week]
wk = f"{subject}_{week}"  # 세션 키 구분용
w.setdefault("summary", "")
st.title(f"{subject} · {week}")
tab1, tab_sum, tab2, tab3, tab4 = st.tabs(
    ["📄 자료·중요 부분", "📑 요약", "📝 퀴즈", "🗒️ 메모장", "📊 기록·오답"]
)

# ---------- 탭1: 자료 / 중요 부분 ----------
with tab1:
    up = st.file_uploader("이번 주 강의 PDF", type="pdf", key=f"pdf_{wk}")
    auto_sum = st.checkbox("저장 후 Gemini로 자동 요약", value=True)
    if up and st.button("PDF 저장"):
        p = pdf_path(subject, week)
        p.write_bytes(up.getvalue())
        w["pdf"] = str(p)
        w["summary"] = ""  # 새 PDF이므로 이전 요약 초기화
        save_db(db)
        st.success("PDF 저장 완료")
        if auto_sum and get_api_key():
            with st.spinner("요약 생성 중... (PDF 크기에 따라 시간이 걸려요)"):
                try:
                    w["summary"] = generate_summary(w, "보통", model)
                    save_db(db)
                    st.success("요약 완료! '📑 요약' 탭에서 확인하세요.")
                except Exception as e:
                    st.error(f"요약 실패: {e}  ('📑 요약' 탭에서 다시 시도할 수 있어요)")
        elif auto_sum:
            st.warning("API 키가 없어 요약을 건너뛰었어요.")
    if w["pdf"] and Path(w["pdf"]).exists():
        st.caption(f"현재 PDF: {Path(w['pdf']).name}")

    st.subheader("중요 부분")
    w["important"] = st.text_area(
        "중요 개념·교수님 강조 내용 등을 적어두세요 (퀴즈에 더 많이 반영돼요)",
        value=w["important"],
        height=180,
    )
    if st.button("중요 부분 저장"):
        save_db(db)
        st.success("저장 완료")

    st.subheader("중요 코드")
    with st.form("snippet_form", clear_on_submit=True):
        c1, c2 = st.columns([3, 1])
        title = c1.text_input("제목 (예: 버블 정렬)")
        lang = c2.selectbox("언어", LANGS)
        code = st.text_area("코드 입력", height=220)
        if st.form_submit_button("코드 추가") and code.strip():
            w["snippets"].append({"title": title or "제목 없음", "lang": lang, "code": code})
            save_db(db)
            st.rerun()
    for i, s in enumerate(w["snippets"]):
        with st.expander(f"{s['title']} ({s['lang']})"):
            st.code(s["code"], language=s["lang"])
            if st.button("삭제", key=f"del_snip_{i}"):
                w["snippets"].pop(i)
                save_db(db)
                st.rerun()

# ---------- 요약 탭 ----------
with tab_sum:
    has_pdf = bool(w["pdf"] and Path(w["pdf"]).exists())
    if not has_pdf:
        st.info("먼저 '자료·중요 부분' 탭에서 PDF를 저장하세요.")
    else:
        c1, c2 = st.columns([3, 1])
        level = c1.radio("요약 분량", ["간단", "보통", "자세히"], index=1, horizontal=True)
        if c2.button("🔄 요약 (재)생성", type="primary", use_container_width=True):
            with st.spinner("Gemini가 요약하는 중..."):
                try:
                    w["summary"] = generate_summary(w, level, model)
                    save_db(db)
                    st.rerun()
                except Exception as e:
                    st.error(f"요약 실패: {e}")
    if w["summary"]:
        st.markdown(w["summary"])
        st.download_button("요약 다운로드(.md)", w["summary"], f"{subject}_{week}_summary.md")
        st.success("요약을 다 읽었다면 '📝 퀴즈' 탭으로 이동하세요 👉")
        with st.expander("✏️ 요약 직접 수정"):
            edited = st.text_area("마크다운으로 수정", value=w["summary"], height=300, key=f"sum_edit_{wk}")
            if st.button("수정 내용 저장"):
                w["summary"] = edited
                save_db(db)
                st.rerun()
    elif has_pdf:
        st.caption("아직 요약이 없어요. 오른쪽 버튼으로 생성하세요.")

# ---------- 탭2: 퀴즈 ----------
with tab2:
    with st.expander("퀴즈 생성 설정", expanded=not w["quiz"]):
        c1, c2, c3 = st.columns(3)
        n_mc = c1.number_input("객관식", 0, 20, 5)
        n_short = c2.number_input("단답/서술", 0, 20, 3)
        n_code = c3.number_input("코드", 0, 10, 2)
        use_pdf = st.checkbox("PDF 내용 포함", value=bool(w["pdf"]))
        if st.button("🎯 퀴즈 생성", type="primary"):
            with st.spinner("Gemini가 문제를 만드는 중..."):
                try:
                    w["quiz"] = generate_quiz(w, n_mc, n_short, n_code, use_pdf, model)
                    save_db(db)
                    st.session_state.pop(f"result_{wk}", None)
                    for k in [k for k in st.session_state if k.startswith(f"q_{wk}_")]:
                        del st.session_state[k]
                    st.rerun()
                except Exception as e:
                    st.error(f"생성 실패: {e}")

    quiz = w["quiz"]
    result = st.session_state.get(f"result_{wk}")
    if quiz:
        answers = []
        for i, q in enumerate(quiz):
            st.markdown(f"**Q{i + 1}. {q['question']}**")
            key = f"q_{wk}_{i}"
            if q["type"] == "mc":
                a = st.radio("답", q["choices"], index=None, key=key, label_visibility="collapsed")
            elif q["type"] == "code":
                a = st.text_area("코드 작성", key=key, height=200, label_visibility="collapsed")
            else:
                a = st.text_area("답 작성", key=key, height=100, label_visibility="collapsed")
            answers.append(a)
            if result:
                r = result[i]
                (st.success if r["correct"] else st.error)("정답!" if r["correct"] else "오답")
                st.caption(r["feedback"])
                if not r["correct"]:
                    if q["type"] == "code":
                        st.code(q["answer"], language="java")
                    else:
                        st.info(f"정답: {q['answer']}")
            st.divider()

        if st.button("✅ 채점하기", type="primary"):
            with st.spinner("채점 중..."):
                try:
                    res = grade(quiz, answers, model)
                except Exception as e:
                    st.error(f"채점 실패: {e}")
                    st.stop()
            st.session_state[f"result_{wk}"] = res
            score = sum(r["correct"] for r in res)
            w["history"].append(
                {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "score": score, "total": len(quiz)}
            )
            for q, r in zip(quiz, res):
                if not r["correct"] and q not in w["wrong"]:
                    w["wrong"].append(q)
            save_db(db)
            st.rerun()
        if result:
            st.metric("점수", f"{sum(r['correct'] for r in result)} / {len(quiz)}")

# ---------- 탭3: 메모장 ----------
with tab3:
    w["memo"] = st.text_area("자유롭게 메모하세요", value=w["memo"], height=400)
    c1, c2 = st.columns(2)
    if c1.button("메모 저장"):
        save_db(db)
        st.success("저장 완료")
    c2.download_button("메모 다운로드(.txt)", w["memo"], f"{subject}_{week}_memo.txt")

# ---------- 탭4: 기록 / 오답 ----------
with tab4:
    st.subheader("점수 기록")
    if w["history"]:
        st.line_chart({"점수(%)": [round(h["score"] / h["total"] * 100) for h in w["history"]]})
        for h in reversed(w["history"]):
            st.write(f"{h['time']} — {h['score']}/{h['total']}")
    else:
        st.caption("아직 기록이 없어요.")

    st.subheader("오답 노트")
    for i, q in enumerate(w["wrong"]):
        with st.expander(q["question"][:60]):
            if q["type"] == "code":
                st.code(q["answer"], language="java")
            else:
                st.write(f"정답: {q['answer']}")
            st.caption(q.get("explanation", ""))
            if st.button("오답 노트에서 삭제", key=f"del_wrong_{i}"):
                w["wrong"].pop(i)
                save_db(db)
                st.rerun()