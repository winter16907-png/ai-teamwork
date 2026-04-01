import streamlit as st
import streamlit_image_select as sti
import os
import requests
import json
import base64
import re
import sqlite3
import io
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_fixed

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
    """初始化資料庫表結構"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  time TEXT, destination TEXT, itinerary TEXT, 
                  images_json TEXT, model TEXT, days INTEGER)''')
    conn.commit()
    conn.close()


def save_new_record_to_db(record):
    """將新行程存入資料庫（初始無圖片）"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO history (time, destination, itinerary, images_json, model, days) VALUES (?,?,?,?,?,?)",
              (record['time'], record['destination'], record['itinerary'],
               json.dumps([]), record['model'], record['days']))
    conn.commit()
    conn.close()


def update_db_images(itinerary_text, images_bytes_list):
    """更新資料庫中的圖片紀錄"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # 將 bytes 轉為 base64 字串以便存入 JSON
    encoded_images = [base64.b64encode(img).decode('utf-8') for img in images_bytes_list]
    c.execute("UPDATE history SET images_json = ? WHERE itinerary = ?",
              (json.dumps(encoded_images), itinerary_text))
    conn.commit()
    conn.close()


def load_history_from_db():
    """從資料庫讀取所有歷史紀錄"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT time, destination, itinerary, images_json, model, days FROM history ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    history = []
    for row in rows:
        # 將 base64 字串轉回 bytes
        decoded_images = [base64.b64decode(img_str) for img_str in json.loads(row[3])]
        history.append({
            "time": row[0], "destination": row[1], "itinerary": row[2],
            "images": decoded_images, "model": row[4], "days": row[5]
        })
    return history


def delete_all_history():
    """清空資料庫"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM history")
    conn.commit()
    conn.close()


# 初始化資料庫
init_db()

st.set_page_config(page_title="AI Trip Planner", layout="wide", page_icon="✈️")

# 初始化時先從資料庫讀取，而非建立空列表
if "history" not in st.session_state:
    st.session_state.history = load_history_from_db()


def reset_budget_callback():
    st.session_state.budget_input = 0


if "budget_input" not in st.session_state:
    st.session_state.budget_input = 20000


# ==========================================
# 2. 核心功能函數
# ==========================================

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_ai_with_fuse(query, model_list, system_prompt):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    for model in model_list:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
                "max_tokens": 8192
            }
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
    clean_p = re.sub(r"^\d+[\s\.\.．:：]*", "", clean_p.strip('"\' '))
    clean_p = clean_p.split("===")[0].strip()
    if not clean_p: return None
    final_p = f"{clean_p}, high-end travel photography, cinematic lighting, ultra-detailed, 8k resolution"
    payload = {
        "model": IMAGE_MODEL, "messages": [{"role": "user", "content": final_p}],
        "modalities": ["image"], "image_config": {"aspect_ratio": aspect_ratio}
    }
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        img_obj = data['choices'][0]['message']['images'][0]
        url = img_obj.get("url") or img_obj.get("image_url", {}).get("url")
        if url and url.startswith("data:"):
            return base64.b64decode(url.split(",", 1)[1])
    except:
        return None

# ==========================================
# 3. 側邊欄 UI
# ==========================================
with st.sidebar:
    st.header("⚙️ Settings")
    model_map = {"Auto-Fuse": [PRIMARY_BRAIN, FUSE_1, FUSE_2], "GPT-4o": [PRIMARY_BRAIN], "DeepSeek V3": [FUSE_1]}
    selected_mode = st.selectbox("AI choice", list(model_map.keys()), key="model_selection_v1")
    lang_options = ["繁體中文", "简体中文", "English", "Other"]
    selected_lang = st.selectbox("Language", lang_options, index=0)
    currency_options = ["CNY", "HKD", "JPY", "MOP", "NTD", "USD", "Other"]
    selected_curr = st.selectbox("Currency", currency_options, index=1)

    if selected_lang == "Other":
        user_lang = st.text_input("Please specify output language (e.g. 日本語)", value="日本語")
    else:
        user_lang = selected_lang

    if selected_curr == "Other":
        # 讓用戶真的能打字輸入貨幣名稱
        user_currency = st.text_input("Please specify currency (e.g. EUR)", value="EUR")
    else:
        user_currency = selected_curr


    st.divider()
    st.header("🕕 Parameters")
    city_from = st.text_input("Departure", value="Hong Kong", key="input_city_from")
    city_to = st.text_input("Destination", key="input_city_to")
    days = st.number_input("Days", min_value=1, value=2, key="input_days")
    theme = st.pills(label="Theme", options=["🌇 Take Photos", "🍕 Enjoy Cuisines", "🎢 Theme Park", "🏛️ Museum"],
                     selection_mode="multi")
    budget = st.number_input(f"Budget ({user_currency})", step=2000, key="budget_input")
    st.button("Predict budget for me", use_container_width=True, on_click=reset_budget_callback)
    st.divider()
    max_images = st.slider("🖼️Images count", 0, 2, 1)
    image_aspect = st.selectbox("Ratio", ["1:1", "16:9", "4:3"], index=1)
    side_submit = st.button("🚀 Start Planning", use_container_width=True, type="primary")
    st.divider()
    if st.button("🗑️ Clear History", use_container_width=True):
        delete_all_history()
        st.session_state.history = []
        st.rerun()

# ==========================================
# 4. 主頁面與邏輯
# ==========================================
col1, col2 = st.columns([0.9, 0.1]) # 調整比例讓問號靠右

with col1:
    st.title("✈️ AI Trip Planner Pro")

with col2:
    with st.popover("❓"):
        tab1, tab2, tab3 = st.tabs(["🚀 Quick Start", "🛠️ Settings", "🔍 Prompt Tips"])
        with tab1:
            st.markdown("# About AI Trip Planner")
            st.markdown("## 1.Introduction")
            st.write(">This is an intelligent travel planning tool based on the OpenRouter API.")
            st.write("The AI trip planner is equipped with top-notch image generation model")
            st.markdown("### 1.1.1 AI text models")
            st.write(">You can choose famous AI models:")
            st.write("1.openai/gpt-oss-120b")
            st.write("2.deepseek/deepseek-v3.2")
            st.write("3.meta-llama/llama-3.3-70b-instruct (standby model)")
            st.markdown("### 1.1.2 AI image model")
            st.markdown(">The AI image model can generate clear, gorgeous images for you")
            st.write("**black-forest-labs/flux.2-pro** (one of the advanced image models around the world)")
            st.markdown("### 1.2 Multiple input method")
            st.write("No matter you use the sidebar or chatbox, AI can generate complete information")
            st.info("It is suggested to use the sidebar to enter information and use the chatbox to enter your needs.")
            st.markdown("### 1.3 Complete planning")
            st.write(">Our models will arrange your flight, hotel, meals, budget and even insurance")
            st.write("**If your destination is dangerous, AI warns you about the risks**")

        with tab2:
            st.markdown("## 2.Slidebar Functions")
            st.write("Multiple functions exist within the user interface and the chatbox")
            st.markdown("### 2.1 Sidebar settings")
            st.markdown("In the sidebar, you can setup the language, AI model, image generation and input your planning")
            st.markdown("#### 2.1.1 AI model setting")
            st.write(">You can select the AI model or just auto-fuse")
            st.markdown("##### Auto-fuse priority:")
            st.write("1.openai/gpt-oss-120b")
            st.write("2.deepseek/deepseek-v3.2")
            st.write("3.meta-llama/llama-3.3-70b-instruct")
            st.markdown("#### 2.1.2 Language setting")
            st.write(">The language below is supported as the output")
            st.write("Traditional Chinese")
            st.write("Simplified Chinese")
            st.write("English")
            st.write("Other (Please enter)")
            st.markdown("#### 2.1.3 Currency setting")
            st.write(">You exchange your money for the local currency and AI provides the exchange rate")
            st.markdown("##### The currencies below can be selected:")
            st.write(">The default currency is HKD")
            st.write("Chinese Yuan (CNY)")
            st.write("Hong Kong Dollar (HKD)")
            st.write("Japanese Yen (JPY)")
            st.write("Macau Pataca (MOP)")
            st.write("New Taiwan Dollar (NTD)")
            st.write("United States Dollar (USD)")
            st.write("Other (Please enter)")
            st.markdown("### 2.2 Input parameters")
            st.write(">Share your initial thoughts on how AI can assist in refining it")
            st.markdown("#### 2.2.1 Departure and Destination")
            st.write(">The default departure point is Hong Kong")
            st.write("You can enter **country names, city names or even airport codes in any language**")
            st.markdown("#### 2.2.2 Duration")
            st.write("The default duration is 3 days")
            st.write("The +/- sign will respectively +/- 1 day")
            st.info("The 3-day means 3 days and 2 nights.")
            st.markdown("2.2.3 Budget")
            st.write(">The default budget is 20000 (the chosen currency), the +/- sign respectively +/- 2000 dollars")
            st.info("The 'Predict budget for me' button will **reset your budget** then\
                    our AI assistants generate it for you.")
            st.markdown("#### 2.3 Image generation setting")
            st.write(">The default setting generates one image, you can use the slider to adjust the number")
            st.markdown("##### The size of images can be set as below:")
            st.write("1.1 (square)")
            st.write("16:9 (recommended)")
            st.write("4:3")

        with tab3:
            st.markdown("## 3.User Interface and Other Functions")
            st.markdown("### 3.1 Historical data")
            st.write(">Everytime our text and each image is generated, the record will be saved")
            st.write("We use sqlite3 to save records, data remains even if you reload the website")
            st.info("Click the 'Clear History' button at the bottom of the slider to delete these records.")
            st.markdown("### ***3.2 Prompt function***")
            st.info("Below are valid prompt formats for your reference.")
            st.image("AImage1.png", caption="Demo of Language, Currency and Destination Setting")
            #st.image
            #st.image()要包括AI自動，語言，其他貨幣，航班代碼；0預算，跨國旅行，圖像生成



if st.session_state.history:
    for item in st.session_state.history:  # 資料庫已按時間排序
        with st.expander(f"{item['time']} - {item['destination']} | 🤖 {item['model'].split('/')[-1]}"):
            clean_hist = re.split(r"===IMAGE_PROMPTS===", item['itinerary'])[0].strip()
            st.markdown(clean_hist)
            if item['images']:
                cols = st.columns(len(item['images']))
                for i, img in enumerate(item['images']): cols[i].image(img)

user_input = st.chat_input("Enter travel details...")

if side_submit or user_input:
    # 如果是從側邊欄觸發且沒有聊天輸入，給一個預設提示或檢查必填項
    final_query_detail = user_input if user_input else "請根據以上設定規劃最佳行程。"

    if not city_to:
        st.error("Please enter a Destination in the sidebar!")
        st.stop()

    SYSTEM_PROMPT = f"""請嚴格遵守以下守則：
    1.框架方面：
    出發地到目的地的簽證/護照等要求
    如果預算不是0，寫出匯率（以{user_currency}兌換目的地的貨幣。例如30,000 HKD ≈ 585,000 JPY），否則寫出以寫出1{user_currency}比幾當地貨幣
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
    預算是0的意思是需要你提供一個舒適合理但不奢華的預算供用戶參考，不代表用戶的預算真的是0（系統設定）
    標題及副標題應該是黑色粗體字體。副標題比普通文字大，標題比副標題大
    必須使用{user_currency}進行所有金額的計算
    42天或以上的行程無需過於詳細，避免token用盡而被截斷"""

    status = st.status("Planning...", expanded=True)
    prompt_instruction = ""
    if max_images > 0:
        placeholders = "\n".join([f"{i + 1}. [描述]" for i in range(max_images)])
        prompt_instruction = f"\n\n請在結束後緊接著輸出此格式：\n===IMAGE_PROMPTS===\n{placeholders}\n===IMAGE_PROMPTS_END==="

    query = f"請用{user_lang}規劃從 {city_from} 到 {city_to} 的 {days} 天行程。主題是{','.join(theme)}預算 {budget}{user_currency}。優先滿足這些需求：{user_input}{prompt_instruction}"

    try:
        raw_itinerary, model_used = call_ai_with_fuse(query, model_map[selected_mode], SYSTEM_PROMPT)

        # 建立紀錄
        new_record = {
            "time": datetime.now().strftime("%m/%d %H:%M"),
            "destination": city_to,
            "itinerary": raw_itinerary,
            "images": [],
            "model": model_used,
            "days": days
        }

        # 1. 存入資料庫
        save_new_record_to_db(new_record)
        # 2. 存入當前 Session 顯示
        st.session_state.history.insert(0, new_record)

        display_itinerary = re.split(r"===IMAGE_PROMPTS===", raw_itinerary)[0].strip()
        status.update(label=f"✅ Planned by {model_used}.", state="complete")
        st.markdown(display_itinerary)

        img_match = re.search(r"===IMAGE_PROMPTS===(.*)", raw_itinerary, re.DOTALL)
        if img_match and max_images > 0:
            prompt_block = img_match.group(1).replace("===IMAGE_PROMPTS_END===", "").strip()
            prompt_list = [l.strip() for l in prompt_block.split("\n") if
                           l.strip() and re.match(r"^\d+[\.\.．\s：:]", l.strip())][:max_images]

            if prompt_list:
                st.subheader("🖼️ Visual Journey")
                img_cols = st.columns(len(prompt_list))
                current_imgs = []
                for i, p in enumerate(prompt_list):
                    with st.spinner(f"Generating image {i + 1}..."):
                        img_bytes = generate_flux_image(p, image_aspect)
                        if img_bytes:
                            current_imgs.append(img_bytes)
                            # 更新 Session State
                            st.session_state.history[0]["images"] = current_imgs
                            # 同步更新資料庫中的圖片
                            update_db_images(raw_itinerary, current_imgs)

                            with img_cols[i]:
                                st.image(img_bytes, use_container_width=True)
                                st.download_button("💾 Download", io.BytesIO(img_bytes), f"trip_{i + 1}.png",
                                                   "image/png", key=f"dl_{hash(p)}")
            else:
                st.warning("AI 輸出標籤但內容格式不正確。")
        elif max_images > 0:
            st.error("AI 完全沒有輸出圖片描述標籤。")
        st.success("Trip saved permanently!")
    except Exception as e:
        st.error(f"Error: {e}")