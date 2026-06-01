## pip install undetected-chromedriver

import os
import random
import string
import time
# 關鍵：改用 undetected_chromedriver 替代原生的 selenium webdriver
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
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


# === 改善點一：模擬真人打字速度的函式 ===
def human_type(element, text):
    element.clear()
    for char in text:
        element.send_keys(char)
        # 每個字元之間隨機延遲 0.08 到 0.25 秒，模擬人類打字的不均勻速度
        time.sleep(random.uniform(0.08, 0.25))


# === ProtonVPN CLI 切換 IP ===
def rotate_vpn_ip():
    print("\n🌐 [IP 切換機制] 正在控制 ProtonVPN 切換節點...")
    try:
        os.system("protonvpn-cli disconnect")
        time.sleep(2)
        os.system("protonvpn-cli connect --fastest")
        print("⏳ [IP 切換機制] 等待網路虛擬通道建立（約 8 秒）...")
        time.sleep(8) 
        print("✅ [IP 切換機制] VPN 已重新連線。")
    except Exception as e:
        print(f"⚠️ ProtonVPN 指令執行失敗: {e}")


# 核心自動化流程
def run_automation():
    # === 改善點二：使用 undetected_chromedriver 的參數設定 ===
    options = uc.ChromeOptions()
    options.add_argument("--incognito")  # 開啟無痕
    
    # 啟動防偵測瀏覽器（這會自動抹除 navigator.webdriver 特徵）
    driver = uc.Chrome(options=options)

    try:
        url = "https://www.61.com.tw/seermain/pre_register?source=MzM1"
        driver.get(url)
        
        # --- 步驟一：輸入隨機米米號（改用擬真打字） ---
        print("正在定位第一個米米號輸入框...")
        mimi_bg_input = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "mimi_bg_mimi"))
        )
        
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", mimi_bg_input)
        time.sleep(random.uniform(0.5, 1.2)) # 模擬人類看見輸入框後的反應時間
        mimi_bg_input.click()
        
        random_mimi = generate_random_digits(8)
        # 調用擬真打字
        human_type(mimi_bg_input, random_mimi)
        print(f"【步驟一成功】已模擬真人打字輸入米米號: {random_mimi}")

        # --- 步驟二：按下第一個「立即登入」 ---
        print("正在尋找第一個立即登入按鈕...")
        mimi_login_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "mimi_bg_login"))
        )
        time.sleep(random.uniform(0.3, 0.8))
        mimi_login_btn.click()
        print("【步驟二成功】已按下立即登入按鈕")
        
        time.sleep(2)  # 等待彈出視窗動畫

        # --- 步驟三：輸入隨機 Email（改用擬真打字） ---
        print("正在尋找彈出表單中的 Email 輸入框...")
        email_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "email"))
        )
        time.sleep(random.uniform(0.4, 0.9))
        random_email = generate_random_email()
        # 調用擬真打字
        human_type(email_input, random_email)
        print(f"【步驟三成功】已模擬真人打字輸入 Email: {random_email}")

        # --- 步驟四：輸入隨機台灣手機號碼（改用擬真打字） ---
        print("正在尋找手機號碼輸入框...")
        phone_input = driver.find_element(By.ID, "phone_number")
        time.sleep(random.uniform(0.2, 0.6))
        random_phone = generate_taiwan_phone()
        # 調用擬真打字
        human_type(phone_input, random_phone)
        print(f"【步驟四成功】已模擬真人打字輸入手機號碼: {random_phone}")

        # --- 步驟五：勾選同意個人資料使用及接收獎勵簡訊 ---
        print("正在勾選同意條款...")
        agree_checkbox = driver.find_element(By.NAME, "userCheck")
        time.sleep(random.uniform(0.3, 0.7))
        if not agree_checkbox.is_selected():
            driver.execute_script("arguments[0].click();", agree_checkbox)
        print("【步驟五成功】已勾選同意條款")

        # --- 步驟六：按下最後的「立即登錄」結束 ---
        print("正在按下最終登錄按鈕...")
        final_register_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "register_btn"))
        )
        time.sleep(random.uniform(0.5, 1.0))
        final_register_btn.click()
        print("【步驟六成功】已按下最終登錄按鈕，單次流程結束")
        
        time.sleep(3)

    except Exception as e:
        print(f"\n❌ 當次執行發生錯誤: {e}")

    finally:
        driver.quit()


# --- 主程式循環控制 ---
if __name__ == "__main__":
    count = 1
    print("====== 終極高隱匿自動化循環腳本已啟動 ======")
    print("====== （欲結束請在終端機按下 Ctrl + C） ======\n")
    
    try:
        while True:
            rotate_vpn_ip()
            print(f"\n▶ 正在執行第 {count} 次流程...")
            run_automation()
            
            delay_time = random.randint(25, 45)
            print(f"第 {count} 次執行完畢。模擬人類隨機延遲 {delay_time} 秒後進行下一次循環...")
            
            for i in range(delay_time, 0, -5):
                print(f"倒數 {i} 秒...")
                time.sleep(min(5, i))
                
            count += 1
            
    except KeyboardInterrupt:
        print("\n\n🛑 偵測到中斷指令（Ctrl + C），自動化循環已安全結束。")