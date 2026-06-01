import random
import string
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 隨機資料生成函式
def generate_random_digits(length=8):
    return "".join(random.choices(string.digits, k=length))

def generate_random_email():
    prefix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    domains = ["gmail.com", "yahoo.com.tw", "hotmail.com"]
    return f"{prefix}@{random.choice(domains)}"

def generate_taiwan_phone():
    return "09" + "".join(random.choices(string.digits, k=8))

# 核心自動化流程
def run_automation():
    # 1. 設定 Chrome 瀏覽器參數（開啟無痕模式）
    chrome_options = Options()
    chrome_options.add_argument("--incognito")  # 啟動無痕模式
    chrome_options.add_argument("--window-size=1920,1080")  # 設定標準視窗大小

    # 啟動瀏覽器
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # 2. 開啟指定網址
        url = "https://www.61.com.tw/seermain/pre_register?source=MzM1"
        driver.get(url)
        
        # --- 步驟一：在指定 input 中輸入隨機生成的八個數字 ---
        print("正在定位第一個米米號輸入框...")
        mimi_bg_input = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "mimi_bg_mimi"))
        )
        
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", mimi_bg_input)
        time.sleep(1)
        mimi_bg_input.click()
        
        random_mimi = generate_random_digits(8)
        mimi_bg_input.clear()
        mimi_bg_input.send_keys(random_mimi)
        print(f"【步驟一成功】已輸入隨機米米號: {random_mimi}")

        # --- 步驟二：按下第一個「立即登入」 ---
        print("正在尋找第一個立即登入按鈕...")
        mimi_login_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "mimi_bg_login"))
        )
        mimi_login_btn.click()
        print("【步驟二成功】已按下立即登入按鈕")
        
        time.sleep(2)  # 等待彈出視窗動畫

        # --- 步驟三：在 id="email" 中輸入隨機產生的 mail ---
        print("正在尋找彈出表單中的 Email 輸入框...")
        email_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "email"))
        )
        random_email = generate_random_email()
        email_input.clear()
        email_input.send_keys(random_email)
        print(f"【步驟三成功】已輸入隨機 Email: {random_email}")

        # --- 步驟四：在 id="phone_number" 中輸入隨機台灣手機號碼 ---
        print("正在尋找手機號碼輸入框...")
        phone_input = driver.find_element(By.ID, "phone_number")
        random_phone = generate_taiwan_phone()
        phone_input.clear()
        phone_input.send_keys(random_phone)
        print(f"【步驟四成功】已輸入隨機手機號碼: {random_phone}")

        # --- 步驟五：勾選同意個人資料使用及接收獎勵簡訊 ---
        print("正在勾選同意條款...")
        agree_checkbox = driver.find_element(By.NAME, "userCheck")
        if not agree_checkbox.is_selected():
            driver.execute_script("arguments[0].click();", agree_checkbox)
        print("【步驟五成功】已勾選同意條款")

        # --- 步驟六：按下最後的「立即登錄」結束 ---
        print("正在按下最終登錄按鈕...")
        final_register_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "register_btn"))
        )
        final_register_btn.click()
        print("【步驟六成功】已按下最終登錄按鈕，單次流程結束")
        
        time.sleep(3)  # 給網頁最後送出資料的時間

    except Exception as e:
        print(f"\n❌ 當次執行發生錯誤: {e}")

    finally:
        # 確保每次流程結束（不論成功或失敗）都會關閉瀏覽器，避免留下大量背景程序
        driver.quit()

# --- 主程式循環控制 ---
if __name__ == "__main__":
    count = 1
    print("====== 自動化腳本已啟動（欲結束請在終端機按下 Ctrl + C） ======")
    
    try:
        while True:
            print(f"\n▶ 正在執行第 {count} 次流程...")
            run_automation()
            
            print(f"第 {count} 次執行完畢。等待 60 秒後進行下一次循環...")
            # 倒數計時提示（可選，方便確認程式有在運作）
            for i in range(60, 0, -5):
                print(f"倒數 {i} 秒...")
                time.sleep(5)
                
            count += 1
            
    except KeyboardInterrupt:
        print("\n\n🛑 偵測到中斷指令（Ctrl + C），自動化循環已安全結束。")