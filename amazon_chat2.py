import os
import re
import time
import torch
import streamlit as st
import warnings
import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import PegasusForConditionalGeneration

warnings.filterwarnings("ignore")

# Gemini setup
os.environ["GOOGLE_API_KEY"] = ""  # 🔐 Add your Gemini API key here
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_chat = genai.GenerativeModel("gemini-2.0-flash-exp").start_chat()

def init_driver():
    options = Options()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--headless=new")  # Optional: run headless
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def extract_asin(url):
    match = re.search(r'/([A-Z0-9]{10})(?:[/?]|$)', url)
    if match:
        return match.group(1)
    raise ValueError("ASIN not found in the URL")

def scrape_product_details(driver, url):
    asin = extract_asin(url)
    product_data = {
        "asin": asin,
        "name": "Unknown",
        "category": "Unknown",
        "description": "Unknown"
    }

    driver.get(url)
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "productTitle")))
        product_data["name"] = driver.find_element(By.ID, "productTitle").text.strip()
        category_elements = driver.find_elements(By.CSS_SELECTOR, 'a.a-link-normal.a-color-tertiary')
        if category_elements:
            product_data["category"] = category_elements[0].text.strip()
        try:
            product_data["description"] = driver.find_element(By.ID, "productDescription").text.strip()
        except:
            try:
                product_data["description"] = driver.find_element(By.ID, "feature-bullets").text.strip()
            except:
                product_data["description"] = "Description not found"
    except:
        st.warning("Failed to extract product details.")
    return product_data

def scrape_reviews(driver, asin, max_pages=2):
    reviews = []
    review_url = f"https://www.amazon.in/product-reviews/{asin}/?pageNumber=1"
    driver.get(review_url)
    time.sleep(2)

    current_page = 1
    while current_page <= max_pages:
        try:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'li[data-hook="review"]')))
        except TimeoutException:
            break

        review_elements = driver.find_elements(By.CSS_SELECTOR, 'li[data-hook="review"]')
        for el in review_elements:
            try:
                rating_html = el.find_element(By.CSS_SELECTOR, 'i[data-hook="review-star-rating"] span.a-icon-alt').get_attribute("outerHTML")
                text = el.find_element(By.CSS_SELECTOR, '[data-hook="review-body"] span').text
                rating_match = re.search(r'(\d+\.\d+|\d+) out of 5 stars', rating_html)
                if rating_match:
                    rating = rating_match.group(1)
                    reviews.append((rating, text))
            except:
                continue

        current_page += 1
        try:
            next_button = driver.find_element(By.CSS_SELECTOR, 'li.a-last a')
            next_button.click()
            time.sleep(2)
        except:
            break
    return reviews

def sentiment(reviews, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    texts = [text for _, text in reviews]
    results = sentiment_pipeline(texts)
    positive, negative = [], []

    for text, result in zip(texts, results):
        if result["label"] == "LABEL_1":
            positive.append(text)
        else:
            negative.append(text)

    return positive, negative

def summarize(text):
    model_name = 'google/pegasus-cnn_dailymail'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

    batch = tokenizer(text, truncation=True, padding='longest', return_tensors="pt").to(device)
    summary_ids = model.generate(**batch)
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    return summary[0].replace("<n>", "\n")

def chat_with_gemini(prompt):
    response = gemini_chat.send_message(prompt)
    return response.text

# ---------------------- Streamlit Interface ----------------------

st.set_page_config(page_title="Amazon Review Analyzer + Gemini Chat", layout="wide")
st.title("📦 Amazon Product Review Analyzer")
tab1, tab2, tab3 = st.tabs(["🕵️ Analyze Product", "📄 Summary", "🤖 Gemini Chatbot"])

with tab1:
    urls_input = st.text_area("Paste Amazon Product URLs (one per line)", height=150)
    model_path = st.text_input("Enter Sentiment Model Path", value=r"C:\models\deberta")
    if st.button("Analyze"):
        urls = urls_input.strip().splitlines()
        if urls:
            with st.spinner("Scraping and analyzing..."):
                driver = init_driver()
                try:
                    product = scrape_product_details(driver, urls[0])
                    reviews = scrape_reviews(driver, product["asin"])
                    if reviews:
                        pos, neg = sentiment(reviews, model_path)
                        st.session_state["product"] = product
                        st.session_state["reviews"] = reviews
                        st.session_state["pos"] = pos
                        st.session_state["neg"] = neg
                        st.success("Done!")
                    else:
                        st.warning("No reviews found.")
                finally:
                    driver.quit()

with tab2:
    if "product" in st.session_state:
        p = st.session_state["product"]
        st.subheader(f"Product: {p['name']}")
        st.write(f"**Category**: {p['category']}")
        st.write(f"**Description**: {p['description']}")

        pos_sum = summarize("\n".join(st.session_state["pos"])) if st.session_state["pos"] else "No positive reviews"
        neg_sum = summarize("\n".join(st.session_state["neg"])) if st.session_state["neg"] else "No negative reviews"

        st.write("✅ **Positive Review Summary:**")
        st.info(pos_sum)
        st.write("❌ **Negative Review Summary:**")
        st.error(neg_sum)
    else:
        st.warning("Please run analysis in Tab 1.")

with tab3:
    st.subheader("Chat with Gemini about the product or summaries")
    user_input = st.text_input("Ask something:")
    if st.button("Send"):
        if user_input:
            response = chat_with_gemini(user_input)
            st.markdown(f"**Gemini:** {response}")
        else:
            st.warning("Please enter a message.")
