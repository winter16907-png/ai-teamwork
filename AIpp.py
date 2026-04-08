import streamlit as st
import os
import requests
import json
import base64
import re
import sqlite3
import io
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_fixed
import uuid
import random as rnd

# ==========================================
# 1. 核心配置與資料庫邏輯
# ==========================================
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
PRIMARY_BRAIN = "openai/gpt-oss-120b"
FUSE_1 = "deepseek/deepseek-v3.2"
FUSE_2 = "meta-llama/llama-3.3-70b-instruct"
IMAGE_MODEL = "black-forest-labs/flux.2-pro"
DB_NAME = "trip_history.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  time TEXT, destination TEXT, itinerary TEXT, 
                  images_json TEXT, model TEXT, days INTEGER,
                  user_id TEXT)''')
    conn.commit()
    conn.close()


def save_new_record_to_db(record):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "INSERT INTO history (time, destination, itinerary, images_json, model, days, user_id) VALUES (?,?,?,?,?,?,?)",
        (record['time'], record['destination'], record['itinerary'],
         json.dumps([]), record['model'], record['days'], record['user_id']))
    conn.commit()
    conn.close()


def update_db_images(itinerary_text, images_bytes_list):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    encoded_images = [base64.b64encode(img).decode('utf-8') for img in images_bytes_list]
    c.execute("UPDATE history SET images_json = ? WHERE itinerary = ?",
              (json.dumps(encoded_images), itinerary_text))
    conn.commit()
    conn.close()


def load_history_from_db(user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "SELECT time, destination, itinerary, images_json, model, days FROM history WHERE user_id = ? ORDER BY id DESC",
        (user_id,))
    rows = c.fetchall()
    conn.close()
    history = []
    for row in rows:
        decoded_images = [base64.b64decode(img_str) for img_str in json.loads(row[3])]
        history.append({
            "time": row[0], "destination": row[1], "itinerary": row[2],
            "images": decoded_images, "model": row[4], "days": row[5]
        })
    return history


def delete_user_history(user_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


# ==========================================
# 2. 輔助工具函數 (背景、圖片、AI)
# ==========================================
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_transparent_bg_via_base64(bin_str, opacity=0.2):
    page_bg_html = f'''
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: linear-gradient(rgba(255, 255, 255, {1 - opacity}), rgba(255, 255, 255, {1 - opacity})),
        url("data:image/jpeg;base64,{bin_str}");
        background-size: cover; background-position: center; background-repeat: no-repeat; background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
    </style>
    '''
    st.markdown(page_bg_html, unsafe_allow_html=True)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_ai_with_fuse(query, model_list, system_prompt):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    for model in model_list:
        try:
            payload = {"model": model,
                       "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                       "max_tokens": 8192}
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload,
                              timeout=60)
            r.raise_for_status()
            return r.json()['choices'][0]['message']['content'], model
        except:
            continue
    raise Exception("All models failed.")


def generate_flux_image(prompt: str, aspect_ratio: str = "16:9"):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    clean_p = re.sub(r"\*\*|\*|📸|圖片描述|AI產圖", "", prompt)
    clean_p = re.sub(r"^\d+[\s\.\.．:：]*", "", clean_p.strip('"\' ')).split("===")[0].strip()
    if not clean_p: return None
    final_p = f"{clean_p}, high-end travel photography, cinematic lighting, ultra-detailed, 8k resolution"
    payload = {"model": IMAGE_MODEL, "messages": [{"role": "user", "content": final_p}], "modalities": ["image"],
               "image_config": {"aspect_ratio": aspect_ratio}}
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        url = r.json()['choices'][0]['message']['images'][0].get("url")
        if url and url.startswith("data:"): return base64.b64decode(url.split(",", 1)[1])
    except:
        return None


# ==========================================
# 3. 初始化與界面設定
# ==========================================
init_db()
st.set_page_config(page_title="AI Trip Planner", layout="wide", page_icon="✈️")

if "user_uuid" not in st.session_state: st.session_state.user_uuid = str(uuid.uuid4())
if "history" not in st.session_state: st.session_state.history = load_history_from_db(st.session_state.user_uuid)
if "dest_state" not in st.session_state: st.session_state.dest_state = ""
if "temp_input" not in st.session_state: st.session_state.temp_input = ""
if "budget_input" not in st.session_state: st.session_state.budget_input = 20000
if "current_output" not in st.session_state: st.session_state.current_output = None

# 固定背景邏輯：避免隨機數導致圖片消失
try:
    # 這裡建議固定一張，或是在 session 內固定隨機數
    if "bg_base64" not in st.session_state:
        bg_file = 'background.png' # 預設檔案
        st.session_state.bg_base64 = get_base64_of_bin_file(bg_file)
    set_transparent_bg_via_base64(st.session_state.bg_base64, opacity=0.4)
except:
    pass

# ==========================================
# 4. 側邊欄 UI
# ==========================================
with st.sidebar:
    page1, page2 = st.tabs(["💻 Settings", "⏰ History"])
    with page1:
        st.header("⚙️ Settings")
        model_map = {"Auto-Fuse": [PRIMARY_BRAIN, FUSE_1, FUSE_2], "GPT-4o (Fast)": [PRIMARY_BRAIN],
                     "DeepSeek V3": [FUSE_1]}
        selected_mode = st.selectbox("AI choice", list(model_map.keys()))
        selected_lang = st.selectbox("Language", ["繁體中文", "简体中文", "English", "Other"])
        user_lang = st.text_input("Language Spec", "日本語") if selected_lang == "Other" else selected_lang
        selected_curr = st.selectbox("Currency", ["CNY", "HKD", "JPY", "MOP", "NTD", "USD", "Other"], index=1)
        user_currency = st.text_input("Currency Spec", "EUR") if selected_curr == "Other" else selected_curr

        st.divider()
        st.header("🕕 Parameters")
        city_from = st.text_input("Departure", value="Hong Kong")
        city_to = st.text_input("Destination", value=st.session_state.dest_state)
        days = st.number_input("Days", min_value=1, value=2)
        theme = st.pills("Theme", ["🌇 Take Photos", "🍕 Enjoy Cuisines", "🎢 Theme Park", "🏛️ Museum", "🛍️ shopping",
                                   "🏖️ vacation"], selection_mode="multi")
        budget = st.number_input(f"Budget ({user_currency})", step=2000, key="budget_input")
        if st.button("Predict budget for me", use_container_width=True):
            st.session_state.budget_input = 0
            st.rerun()

        st.divider()
        max_images = st.slider("🖼️Images", 0, 2, 1)
        image_aspect = st.selectbox("Ratio", ["1:1", "16:9", "4:3"], index=1)
        side_submit = st.button("🚀 Start Planning", use_container_width=True, type="primary")

    with page2:
        st.header("📜 History")
        if st.session_state.history:
            for item in st.session_state.history:
                with st.expander(f"{item['time']} - {item['destination']}"):
                    st.markdown(re.split(r"===IMAGE_PROMPTS===", item['itinerary'])[0].strip())
                    if item['images']:
                        cols = st.columns(2)
                        for i, img in enumerate(item['images']):
                            cols[i % 2].image(img, use_container_width=True)
        else:
            st.info("No history yet.")
        if st.button("🗑️ Clear History", use_container_width=True):
            delete_user_history(st.session_state.user_uuid)
            st.session_state.history = [];
            st.rerun()

# ==========================================
# 5. 主頁面佈局
# ==========================================
# --- 標題與 CSS ---
st.markdown("""
<div style="background: rgba(255, 255, 255, 0.15); backdrop-filter: blur(20px); border-radius: 20px; padding: 30px; border: 1px solid rgba(255,255,255,0.2);">
    <h1>✈️ AI Trip Planner Pro</h1>
    <p>Developed by kalokwong6's team | ℹ️ AI results may be inaccurate.</p>
</div>
""", unsafe_allow_html=True)

try:
    img_bg = get_base64_of_bin_file(
        f'background{rnd.randint(1, 2)}.png' if rnd.randint(1, 2) == 2 else 'background.png')
    set_transparent_bg_via_base64(img_bg, opacity=0.4)
except:
    pass

# --- 核心容器：確保 AI 輸出在按鈕上方 ---
ai_output_area = st.container()
with ai_output_area:
    if st.session_state.current_output:
        res = st.session_state.current_output
        st.markdown(res['text'])
        if res['images']:
            cols = st.columns(len(res['images']))
            for idx, img in enumerate(res['images']):
                cols[idx].image(img, use_container_width=True)
st.divider()

# --- 快捷按鈕 (Hot Spots) ---
st.write("💡 **Hot spots**")


def set_prompt(prompt, dest):
    st.session_state.temp_input = prompt
    st.session_state.dest_state = dest
    st.rerun()


h_cols = st.columns(4)
spots = [
    ("🇯🇵 Tokyo & Osaka", "Tokyo and Osaka one-week trip with Shinkansen.", "HND"),
    ("🇪🇺 Europe Trip", "England, France, Italy, Germany for a month.", "LHR"),
    ("🇱🇻 Baltic States", "Three Baltic states culture trip.", "RIX"),
    ("🏝️ Hawaii Tour", "Hawaii island-hopping and volcanoes.", "HNL")
]
for i, (lab, p, d) in enumerate(spots):
    if h_cols[i].button(lab, use_container_width=True): set_prompt(p, d)

# --- 聊天輸入框 ---
user_input = st.chat_input("Enter travel details...")

# ==========================================
# 6. 執行邏輯
# ==========================================
if side_submit or user_input or st.session_state.temp_input:
    final_detail = user_input if user_input else st.session_state.temp_input
    st.session_state.temp_input = ""
    target_city = city_to if city_to else st.session_state.dest_state

    if not target_city:
        st.error("Please enter a Destination!");
        st.stop()

    with ai_output_area:
        status = st.status("Planning your dream trip...", expanded=True)

        # ... (SYSTEM_PROMPT 與 query 保持不變) ...

    if not target_city:
        st.error("Please enter a Destination!");
        st.stop()

    with ai_output_area:
        status = st.status("Planning your dream trip...", expanded=True)

        SYSTEM_PROMPT = f"""請嚴格遵守以下守則：
                1.框架方面：
                出發地到目的地的簽證/護照等要求
                如果預算不是0，寫出匯率（以{user_currency}兌換目的地的貨幣。例如30,000 HKD ≈ 585,000 JPY），否則寫出以寫出1{user_currency}比幾當地貨幣，必須標註「The exchange rate may vary, and the actual exchange rate will be based on the bank's rate.」
                簡單寫出用戶的預算是寬鬆/合理/緊湊，附上1-3句解釋
                酒店（建議推薦住宿費用包含早餐的酒店，附上價格，位置是否方便，評價等）
                機票（優先選擇不需轉機的航班，附上起飛日期，來回機票價格，航空公司等，如要轉機，列出所有中轉站），必須標註「The flight number is for reference only; please refer to the actual booking.」
                行程（第幾天的什麼時間去哪裡大約花多少錢做什麼，包括午晚餐）
                旅遊保險（哪一家公司以什麼價格提供什麼服務）
                預算分配，必須標註「The price is based on AI simulation results; please refer to actual consumption for accurate figures.」
                注意事項（如充電插頭差異）
                圖片生成（如有）
                2.安全方面：
                如果目的地有戰亂，災害，衛生和政治之類的安全問題，請警告用戶不要前往
                3.其他方面：
                天數必須嚴格等於用戶設定的數字，出發日為第一天，回程日為最後一天。
                如果用戶語言為Auto，根據用戶的輸入語言判斷輸出語言
                如果用戶的預算過於緊湊，幾乎不可能，可以適度增加預算；但是如果用戶的時間過於緊湊，優先考慮縮減行程而不是增加旅行日期
                預算是0的意思是需要你提供一個舒適合理但不奢華的預算供用戶參考，用戶真正的預算並不是0
                標題及副標題應該是黑色粗體字體。副標題比普通文字大，標題比副標題大
                必須使用{user_currency}進行所有金額的計算
                如有必要，請提醒用戶有關時差和夏令時/冬令時的消息
                56天或以上的行程無需過於詳細，避免token用盡而被截斷
                """

        prompt_instruction = ""
        if max_images > 0:
            prompt_instruction = f"\n\n===IMAGE_PROMPTS===\n" + "\n".join(
                [f"{i + 1}. [描述]" for i in range(max_images)]) + "\n===IMAGE_PROMPTS_END==="

        query = f"從 {city_from} 到 {target_city} 的 {days} 天行程。主題:{','.join(theme)}，需求：{final_detail}{prompt_instruction}"

        try:
            raw_itinerary, model_used = call_ai_with_fuse(query, model_map[selected_mode], SYSTEM_PROMPT)
            display_text = re.split(r"===IMAGE_PROMPTS===", raw_itinerary)[0].strip()

            # 準備存入狀態
            st.session_state.current_output = {"text": display_text, "images": []}

            # 存入資料庫
            rec = {"time": datetime.now().strftime("%m/%d %H:%M"), "destination": target_city,
                   "itinerary": raw_itinerary, "images": [], "model": model_used, "days": days,
                   "user_id": st.session_state.user_uuid}
            save_new_record_to_db(rec)

            status.write(display_text)  # 在 status 內也顯示進度

            # 圖片生成邏輯
            img_match = re.search(r"===IMAGE_PROMPTS===(.*)===IMAGE_PROMPTS_END===", raw_itinerary, re.DOTALL)
            generated_imgs = []
            if img_match and max_images > 0:
                p_list = [l.strip() for l in img_match.group(1).split("\n") if re.match(r"^\d", l.strip())][:max_images]
                for p in p_list:
                    img_b = generate_flux_image(p, image_aspect)
                    if img_b:
                        generated_imgs.append(img_b)

                # 更新圖片到狀態與資料庫
                st.session_state.current_output["images"] = generated_imgs
                update_db_images(raw_itinerary, generated_imgs)

            status.update(label=f"✅ Planned by {model_used}.", state="complete")

            # 重要：更新歷史紀錄列表
            st.session_state.history = load_history_from_db(st.session_state.user_uuid)

            # 手動觸發一次重新渲染以更新側邊欄，此時因為 current_output 已有值，結果不會不見
            st.rerun()

        except Exception as e:
            st.error(f"❌ Planning failed: {str(e)}")