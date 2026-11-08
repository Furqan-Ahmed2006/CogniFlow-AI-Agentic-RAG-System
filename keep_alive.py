import os
import time
from playwright.sync_api import sync_playwright

APP_URL = os.getenv("STREAMLIT_APP_URL", "https://your-cognirag-ai-app-url.streamlit.app")

def ping_streamlit_app():
    print(f"🚀 Launching headless browser to ping: {APP_URL}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            page.goto(APP_URL, timeout=60000, wait_until="networkidle")
            time.sleep(5) 
            
            
            wake_button = page.query_selector("button:has-text('Yes, get this app back up!')")
            if wake_button:
                print("⚡ App was asleep. Clicking Wake Up button...")
                wake_button.click()
                time.sleep(15)
            

            print("✅ CogniRAG-AI Keep-Alive Ping Successful!")
            
        except Exception as e:
            print(f"❌ Error pinging Streamlit app: {str(e)}")
            
        finally:
            browser.close()

if __name__ == "__main__":
    ping_streamlit_app()