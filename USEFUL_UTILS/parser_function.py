import time
from USEFUL_UTILS.sql_functions import SqlFunctions
SQL = SqlFunctions()

from selenium.webdriver.common.by import By
from seleniumwire import webdriver
opts = webdriver.ChromeOptions()

from selenium.webdriver.chrome.service import Service
service = Service()


class ParserFunction():
    def __init__(self):
        self.headless = True

        proxy_df = SQL.uploadSqlTab('proxy_base')
        self.proxy_list = list(proxy_df['proxy'].values)
        self.proxy_idx = 0

    def getDriver(self, with_proxy=True):

        if with_proxy:
            my_proxy_list = self.proxy_list
            proxy = my_proxy_list[self.proxy_idx]
            print(f"proxy = {proxy}, proxy_idx = {self.proxy_idx}")

            self.proxy_idx += 1
            if self.proxy_idx == len(my_proxy_list):
                self.proxy_idx = 0

            options = {
                'proxy': {'http': f"http://{proxy}", 'https': f"https://{proxy}",
                          'no_proxy': 'localhost,127.0.0.1'}
            }


        opts.add_argument("--start-maximized")
        opts.add_argument('window-size=2560,1440')
        if self.headless: opts.add_argument("--headless=new") #безголовый режим
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-features=VizDisplayCompositor")

        if with_proxy:
            driver = webdriver.Chrome(service=service, options=opts, seleniumwire_options=options)

        else:
            driver = webdriver.Chrome(service=service, options=opts)

        return driver

    def get_AddBlock_Driver(self, with_proxy=True):
        opts = webdriver.ChromeOptions()

        # === Твои обычные аргументы ===
        opts.add_argument("--start-maximized")
        opts.add_argument('window-size=2560,1440')
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-features=VizDisplayCompositor")
        if self.headless:
            opts.add_argument("--headless=new")

        # === КЛЮЧЕВЫЕ аргументы для подавления рекламы ===
        opts.add_argument("--mute-audio")  # звук рекламы
        opts.add_argument("--disable-features=TranslateUI")
        opts.add_argument("--disable-extensions")  # чтобы ничего не мешало
        opts.add_argument("--disable-notifications")
        opts.add_argument("--disable-popup-blocking")

        # === Прокси через selenium-wire (если нужен) ===
        seleniumwire_options = None
        if with_proxy:
            my_proxy_list = self.proxy_list
            proxy = my_proxy_list[self.proxy_idx]
            print(f"proxy = {proxy}, proxy_idx = {self.proxy_idx}")

            self.proxy_idx += 1
            if self.proxy_idx >= len(my_proxy_list):
                self.proxy_idx = 0

            seleniumwire_options = {
                'proxy': {
                    'http': f'http://{proxy}',
                    'https': f'https://{proxy}',
                    'no_proxy': 'localhost,127.0.0.1'
                }
            }

        # === Создаём драйвер ===
        if with_proxy:
            driver = webdriver.Chrome(
                service=service,
                options=opts,
                seleniumwire_options=seleniumwire_options
            )
        else:
            driver = webdriver.Chrome(service=service, options=opts)

        # =================================================================
        # ВСЁ, ЧТО НУЖНО ДЛЯ УНИЧТОЖЕНИЯ РЕКЛАМЫ (Rutube + caramel-fadeInBox)
        # =================================================================

        # 1. Блокируем домены на уровне сети (самое мощное)
        driver.execute_cdp_cmd("Network.enable", {})
        driver.execute_cdp_cmd("Network.setBlockedURLs", {
            "urls": [
                "*rutube.ru*",
                "*bl.rutube.ru*",
                "*st.rutube.ru*",
                "*cdn.rutube.ru*",
                "*rutube.cdnvideo.ru*",
                "*caramel-ads.*",  # если реклама от Caramel
                "*yandex.ru/ads*",
                "*an.yandex.ru*",
                "*adfox.ru*"
            ]
        })

        # 2. Встраиваем JS-скрипт ДО загрузки любой страницы
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
            // Запрещаем попапы и рекламу на корню
            window.open = () => null;
            window.alert = () => false;

            // Удаляем любой caramel-fadeInBox и Rutube сразу при появлении
            new MutationObserver((mutations) => {
                for (const m of mutations) {
                    for (const node of m.addedNodes) {
                        if (node.nodeType !== 1) continue;
                        const cls = node.classList;
                        if (cls && (
                            cls.contains('caramel-fadeInBox') ||
                            cls.contains('modal') ||
                            cls.contains('popup') ||
                            cls.contains('advert') ||
                            node.querySelector && node.querySelector('.caramel-fadeInBox')
                        )) {
                            node.remove();
                        }
                        if (node.tagName === 'IFRAME' && node.src && node.src.includes('rutube')) node.remove();
                    }
                }
            }).observe(document.documentElement, { childList: true, subtree: true });

            // Блокируем fetch/XHR к рекламе
            const block = url => url.includes('rutube') || url.includes('caramel') || url.includes('ads');
            const ofetch = window.fetch;
            window.fetch = (...a) => block(a[0]) ? Promise.resolve(new Response('')) : ofetch(...a);
            """
        })

        # 3. (Опционально) Авто-клик по крестику через 3 сек, если что-то проскочило
        driver.execute_script("""
            setTimeout(() => {
                const close = document.evaluate("//div[contains(@class,'close') or contains(text(),'×') or contains(text(),'Пропустить')]", 
                    document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                if (close) close.click();
            }, 3000);
        """)

        return driver


    def driverIsAlive(self, driver):
        try:
            driver.current_url
            return True
        except:
            return False

    def driverCloseQuit(self, driver):
        if self.driverIsAlive(driver):
            driver.close()
            time.sleep(.2)
            driver.quit()
            time.sleep(.3)

    def directlyScrollIntoView(self, driver, element):
        driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'})", element)
        time.sleep(.4)
        return driver

    def scrollClick(self, driver, element):
        driver = self.directlyScrollIntoView(driver, element)
        element.click()
        return driver

    def xpathFindText(self, driver, ss):
        #поиск элемента по тексту
        try:
            return driver.find_element(By.XPATH, f"//*[contains(text(), '{ss}')]")
        except Exception as e:
            print(e)
            return False







