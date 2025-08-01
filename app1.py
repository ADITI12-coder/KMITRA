import asyncio
import nest_asyncio
import json
import re
import os
import requests
import torch
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from pyngrok import ngrok
from flask_cors import CORS
from playwright.async_api import async_playwright

# New imports for RAG with FAISS and Gemini
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import time  # To prevent API rate limits

# ---------------------- Configuration ----------------------
nest_asyncio.apply()
device = "cuda" if torch.cuda.is_available() else "cpu"
HEADERS = {"User-Agent": "Mozilla/5.0"}
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "2uE2st3psI5jXvZHas2LFpuL8V1_4uAUDZ3keo7e6rnmNmwAg")

# URLs and credentials for attendance, results, timetable scraping remain unchanged
LOGIN_URL = "http://kmit-netra.teleuniv.in/"
RESULTS_URL = "http://kmit-netra.teleuniv.in/student/results"
TIME_TABLE_URL = "http://kmit-netra.teleuniv.in/student/time-table"
MOBILE_NUMBER = "8500616722"
PASSWORD = "Kmit123$"

# ---------------------- New RAG Implementation using FAISS & Gemini ----------------------
# Load FAISS index and metadata
try:
    index = faiss.read_index("faiss_index.bin")
    with open("faiss_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    texts = metadata["texts"]
    urls = metadata["urls"]
    print("FAISS index and metadata loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS index or metadata: {e}")
    texts = []
    urls = []

# Load Sentence Transformer for query embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Configure Gemini API – Replace with your actual key
genai.configure(api_key="AIzaSyBiOCbcv1eqK0eKFQdqYH3EUMBGQdYNWdY")
model = genai.GenerativeModel("gemini-1.5-flash")

def retrieve_top_7_and_generate_response(query):
    """Retrieve the top 7 relevant text chunks from FAISS and generate a response via Gemini AI."""
    # Embed the query
    query_embedding = embed_model.encode([query], convert_to_numpy=True)

    # Search FAISS for relevant results (search over all texts)
    distances, indices = index.search(query_embedding, len(texts))
    
    # Retrieve top 7 most relevant results
    retrieved_texts = [texts[i] for i in indices[0][:30]]
    retrieved_context = "\n".join(retrieved_texts)
    
    # Format prompt for Gemini
    prompt = f"""
        You are an expert AI assistant. Based solely on the following context, provide a concise, accurate, and direct answer to the question.
         Do not repeat or include any extraneous information from the context—focus only on what is necessary for the answer.

    Context:
    {retrieved_context}

    Question: {query}

    Answer:"""

    
    # Get response from Gemini
    response = model.generate_content(prompt)
    return response.text

# ---------------------- Existing Web Scraping Functions ----------------------
async def get_percentage(cell):
    svg = cell.find('svg', class_='ant-progress-circle')
    if not svg:
        return None
    circles = svg.find_all('circle', class_='ant-progress-circle-path')
    if not circles:
        return None
    for circle in circles:
        style = circle.get('style', '')
        dasharray_match = re.search(r'stroke-dasharray:\s*([\d.]+)', style)
        dashoffset_match = re.search(r'stroke-dashoffset:\s*([\d.]+)', style)
        if dasharray_match and dashoffset_match:
            try:
                dasharray = float(dasharray_match.group(1))
                dashoffset = float(dashoffset_match.group(1))
                if dasharray > 0:
                    return round((1 - (dashoffset / dasharray)) * 100, 1)
            except Exception as e:
                print(f"Error parsing circle: {e}")
                return None
    return None
async def fetch_attendance(mobile_number, password):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(LOGIN_URL)

        # Fill login credentials
        await page.fill(".ant-input", mobile_number)
        await page.fill(".ant-input-password input", password)
        await page.click(".ant-btn-primary")
        
        # Wait for an element that confirms a successful login
        try:
            # Increase timeout to 30 seconds and wait for one of two possible selectors
            await page.wait_for_selector("span.ninjadash-nav-actions__author--name, div.profile-header", timeout=30000)
        except Exception as e:
            print("Login might have failed or took too long:", e)
            await browser.close()
            return {"error": "Login failed or timed out."}

        # Navigate to the attendance page after confirming login success
        await page.goto("http://kmit-netra.teleuniv.in/student/attendance")
        await page.wait_for_timeout(5000)

        overall_header = page.locator('h4:has-text("Overall")')
        if await overall_header.count():
            await overall_header.click()
            await page.wait_for_timeout(3000)
        html_content = await page.content()
        soup = BeautifulSoup(html_content, "html.parser")
        student_name = soup.find("span", class_="ninjadash-nav-actions__author--name")
        if student_name:
            print(f"\n✅ Student: {student_name.text.strip()}")

        attendance_table = soup.find('table')
        attendance_data = []
        if attendance_table:
            tbody = attendance_table.find('tbody', class_='ant-table-tbody')
            if tbody:
                rows = tbody.find_all('tr', class_='ant-table-row')
                for row in rows:
                    cells = row.find_all('td', class_='ant-table-cell')
                    if len(cells) >= 3:
                        subject = cells[0].get_text(strip=True)
                        theory = await get_percentage(cells[1])
                        practical = await get_percentage(cells[2])
                        attendance_data.append({
                            'subject': subject,
                            'theory': theory,
                            'practical': practical
                        })

        overall_progress = soup.find('div', class_='ant-progress-bg')
        overall_percent = None
        if overall_progress:
            overall_style = overall_progress.get('style', '')
            try:
                overall_percent = overall_style.split('width:')[-1].split('%')[0].strip()
            except IndexError:
                print("❌ Could not extract overall attendance percentage.")
        await browser.close()
        return {
            "student_name": student_name.text.strip() if student_name else "Unknown",
            "attendance_data": attendance_data,
            "overall_percent": overall_percent
        }



def fetch_news_bulletins():
    url = "https://kmit.in/"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        bulletins = []
        section = soup.find("div", class_="tab-content newsandbulletinstabcontent border")
        if not section:
            return []
        for news in section.find_all("li", class_="news-item"):
            text = news.get_text(strip=True)
            link_tag = news.find("a")
            link = link_tag['href'] if link_tag and 'href' in link_tag.attrs else None
            bulletins.append(f"{text} - {link}" if link else text)
        return bulletins
    except requests.exceptions.RequestException:
        return []

def fetch_exam_timetables():
    url = "https://kmit.in/examination/exam.php"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        timetables = []
        section = soup.find("div", id="Examtimetable")
        if not section:
            return []
        rows = section.find("table", class_="table-striped").find("tbody").find_all("tr")[:10]
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            text = cols[0].get_text(strip=True)
            date_posted = cols[1].get_text(strip=True)
            timetables.append(f"{text} (Posted on {date_posted})")
        return timetables
    except requests.exceptions.RequestException:
        return []

def fetch_exam_notifications():
    url = "https://kmit.in/examination/exam.php"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        notifications = []
        section = soup.find("div", id="Examnotification")
        if not section:
            return []
        table = section.find("table", class_="table-striped")
        if not table:
            return []
        rows = table.find("tbody").find_all("tr")[:10]
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            link_tag = cols[0].find("a")
            text = link_tag.get_text(strip=True) if link_tag else cols[0].get_text(strip=True)
            href = (link_tag.get('data-whatever') or link_tag.get('href')) if link_tag else None
            pdf_link = f"https://kmit.in{href}" if href else None
            date_posted = cols[1].get_text(strip=True)
            entry = f"{text} (Posted on {date_posted})"
            if pdf_link:
                entry += f" - {pdf_link}"
            notifications.append(entry)
        return notifications
    except requests.exceptions.RequestException:
        return []

# ---------------------- Flask App ----------------------
app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    try:
        data = request.json
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Use new RAG implementation to generate a response.
        answer = retrieve_top_7_and_generate_response(query)
        return jsonify({
            "query": query,
            "response": answer
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/attendance", methods=["POST"])
def attendance():
    data = request.json
    mobile_number = data.get("mobile_number")
    password = data.get("password")
    if not mobile_number or not password:
        return jsonify({"error": "Mobile number and password are required"}), 400
    attendance_data = asyncio.run(fetch_attendance(mobile_number, password))
    return jsonify(attendance_data)

@app.route("/timetable", methods=["POST"])
def timetable():
    data = request.json
    mobile_number = data.get("mobile_number")
    password = data.get("password")
    if not mobile_number or not password:
        return jsonify({"error": "Mobile number and password are required"}), 400
    timetable_data = asyncio.run(fetch_timetable_data(mobile_number, password))
    return jsonify(timetable_data)

@app.route("/results", methods=["POST"])
def results():
    try:
        results_data = asyncio.run(fetch_results_data())
        return jsonify(results_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------- Server Setup ----------------------
if __name__ == "__main__":
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    public_url = ngrok.connect(5000).public_url
    print(f" * Ngrok tunnel running at: {public_url}")
    app.run(port=5000)