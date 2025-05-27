import os
import re
import time
import torch
import streamlit as st
import warnings
import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers import PegasusForConditionalGeneration
from selenium.webdriver.chrome.service import Service

warnings.filterwarnings("ignore")

# Gemini setup (safe: API key not hardcoded)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_chat = genai.GenerativeModel("gemini-2.0-flash-exp").start_chat()

# Extract ASIN from Amazon URL
def extract_asin(url):
    match = re.search(r'/([A-Z0-9]{10})(?:[/?]|$)', url)
    if match:
        return match.group(1)
    raise ValueError("ASIN not found in the URL")

# Scrape reviews using Selenium + webdriver_manager
def scrape_reviews_from_urls(product_urls, max_pages=2):
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-setuid-sandbox")
    driver_version = "120.0.6099.224"  
    service = Service(ChromeDriverManager(driver_version = driver_version).install())
    driver = webdriver.Chrome(service=service, options=options)
    results = []

    try:
        for url in product_urls:
            asin = extract_asin(url)
            product_data = {
                "name": "Unknown",
                "category": "Unknown",
                "description": "Unknown",
                "reviews": []
            }

            try:
                driver.get(url)
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "productTitle")))
                product_data["name"] = driver.find_element(By.ID, "productTitle").text.strip()
                category_elements = driver.find_elements(By.CSS_SELECTOR, 'a.a-link-normal.a-color-tertiary')
                product_data["category"] = category_elements[0].text.strip() if category_elements else "Unknown"

                try:
                    desc_elem = driver.find_element(By.ID, "productDescription")
                    product_data["description"] = desc_elem.text.strip()
                except:
                    try:
                        bullet_points = driver.find_element(By.ID, "feature-bullets")
                        product_data["description"] = bullet_points.text.strip()
                    except:
                        product_data["description"] = "Description not found"
            except:
                continue

            # Review scraping
            review_url = f"https://www.amazon.in/product-reviews/{asin}/?pageNumber=1"
            driver.get(review_url)
            time.sleep(3)

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
                            product_data["reviews"].append((rating, text))
                    except:
                        continue

                current_page += 1
                try:
                    next_button = driver.find_element(By.CSS_SELECTOR, 'li.a-last a')
                    next_button.click()
                    time.sleep(2)
                except:
                    break

            results.append(product_data)
    finally:
        driver.quit()

    return results

# Run local sentiment model
def sentiment(data, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    reviews = [review[1] for review in data[0]['reviews']]
    results = sentiment_pipeline(reviews)
    positive, negative = [], []

    for review, result in zip(reviews, results):
        if result["label"] == "LABEL_1":
            positive.append(review)
        else:
            negative.append(review)

    return positive, negative

# Summarize using Pegasus
def summarize(text):
    model_name = 'google/pegasus-cnn_dailymail'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

    batch = tokenizer(text, truncation=True, padding='longest', return_tensors="pt").to(device)
    summary_ids = model.generate(**batch)
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    return summary[0].replace("<n>", "\n")

# Chat with Gemini
def chat_with_gemini(prompt):
    response = gemini_chat.send_message(prompt)
    return response.text

# Streamlit UI
st.set_page_config(page_title="Amazon Review Analyzer + Gemini Chat", layout="wide")
st.title("📦 Amazon Product Review Analyzer")

tab1, tab2, tab3 = st.tabs(["🕵️ Analyze Product", "📄 Summary", "🤖 Gemini Chatbot"])

with tab1:
    urls_input = st.text_area("Paste Amazon Product URLs (one per line)", height=150)
    model_path = st.text_input("Enter Sentiment Model Path", value=r"C:\models\deberta")
    if st.button("Analyze"):
        urls = urls_input.strip().splitlines()
        with st.spinner("Scraping and analyzing..."):
            product_data = scrape_reviews_from_urls(urls)
            if product_data and product_data[0]["reviews"]:
                pos, neg = sentiment(product_data, model_path)
                st.session_state["product"] = product_data[0]
                st.session_state["pos"], st.session_state["neg"] = pos, neg
                st.success("Done!")

with tab2:
    if "product" in st.session_state:
        st.subheader(f"Product: {st.session_state['product']['name']}")
        st.write(f"**Category**: {st.session_state['product']['category']}")
        st.write(f"**Description**: {st.session_state['product']['description']}")

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
