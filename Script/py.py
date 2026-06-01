import os
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


# === 透過 ProtonVPN CLI 切換 IP ===
def rotate_vpn_ip():
    print("\n🌐 [IP 切換機制] 正在控制 ProtonVPN 切換節點...")
    try:
        # 1. 斷開目前的 VPN 連線
        print("斷開當前 VPN...")
        os.system("protonvpn-cli disconnect")
        time.sleep(2)
        
        # 2. 隨機連線到目前最快的免費伺服器 (通常在美國、荷蘭或日本之間切換)
        print("重新連線至最快的免費伺服器...")
        os.system("protonvpn-cli connect --fastest")
        
        # 3. 等待安全隧道建立與 IP 分配
        print("⏳ [IP 切換機制] 等待網路虛擬通道建立（約 8 秒）...")
        time.sleep(8) 
        print("✅ [IP 切換機制] VPN 已重新連線，順利取得新國家/地區 IP。")
    except Exception as e:
        print(f"⚠️ ProtonVPN 指令執行失敗，請確認是否已安裝 CLI 工具並加入環境變數: {e}")


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
        # 關閉瀏覽器，準備下一次更換全新網路環境
        driver.quit()


# --- 主程式循環控制 ---
if __name__ == "__main__":
    count = 1
    print("====== ProtonVPN 動態變更 IP 循環腳本已啟動 ======")
    print("====== （欲結束請在終端機按下 Ctrl + C） ======\n")
    
    try:
        while True:
            # 1. 每次迴圈開頭先切換 VPN 節點以更換外部公網 IP
            rotate_vpn_ip()
            
            # 2. 執行網頁自動化填表
            print(f"\n▶ 正在執行第 {count} 次流程...")
            run_automation()
            
            # 3. 執行完畢後的模擬人類隨機冷卻等待
            delay_time = random.randint(25, 45)
            print(f"第 {count} 次執行完畢。隨機延遲 {delay_time} 秒後進行下一次循環...")
            
            for i in range(delay_time, 0, -5):
                print(f"倒數 {i} 秒...")
                time.sleep(min(5, i))
                
            count += 1
            
    except KeyboardInterrupt:
        print("\n\n🛑 偵測到中斷指令（Ctrl + C），自動化循環已安全結束。")