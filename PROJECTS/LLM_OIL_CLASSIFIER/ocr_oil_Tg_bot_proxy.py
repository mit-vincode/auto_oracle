# ocr_oil_Tg_bot.py
# Telegram-бот для RAG-системы по автохимии
# aiogram 3.x + llama.cpp + FAISS + прокси для России

import pandas as pd
import os

from bootstrap import *

import logging, platform
import asyncio
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import Command
import html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

# ← НОВОЕ: сессия с прокси
from aiogram.client.session.aiohttp import AiohttpSession

from PROJECTS.LLM_OIL_CLASSIFIER.ocr_QA import answerGenerate


TEST_mode = True

# Загружаем переменные окружения
load_dotenv(find_dotenv())

if TEST_mode and (platform.system() == "Darwin"):
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") #LavrGPT
    print(f"TEST_mode = {TEST_mode}")
else:
    BOT_TOKEN = os.getenv("TELEGRAM_MaxGPT_TOKEN") #@Auto_Oracle_Bot

# === НАСТРОЙКИ ПРОКСИ (РАБОЧИЙ В РОССИИ) ===
PROXY_URL = "http://M3MRa2:Q1Pp2Y@177.234.136.58:8000"   # ← ваш проверенный прокси

# Создаём сессию с прокси
session = AiohttpSession(proxy=PROXY_URL)

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("reinwell_ruseff_liquimoly")

# Создаём бота с прокси-сессией
default_properties = DefaultBotProperties(parse_mode=ParseMode.HTML)
bot = Bot(token=BOT_TOKEN, session=session, default=default_properties)

dp = Dispatcher()


@dp.message(Command("start", "help"))
async def cmd_start(message: Message):
    await message.answer(
        "Привет! Я — ИИ-эксперт по маслам и автохимии\n\n"
        "Умею подбирать автомобильное масло и жидкости, консультирую по заправочным объёмам\n"
        "Задавай любой вопрос, например:\n\n"
        "• Opel astra h 1.6 2018 масло моторное\n"
        "• Nissan X-Trail 2012 масло в раздатку\n"
        "• Zeekr 001 антифриз в батарею\n"
        "• VOYAH FREE 1.5 л масло редуктора\n"
        "• Как работает воздушный фильтр?\n"
        "• Как можно самостоятельно промыть радиатор автомобиля?\n\n"
        "Тестовая модель (MVP), ответ ~ 30–90 сек\n\n"
        "По вопросам: @mit_aDat",
        disable_web_page_preview=True,
    )


@dp.message(F.text)
async def handle_question(message: Message):
    question = (message.text or "").strip()
    if not question:
        await message.answer("Пустое сообщение.")
        return

    user_id = message.from_user.id
    username = message.from_user.username or "unknown"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"[{stamp}] Вопрос от @{username} (id={user_id}): {question}")

    # Показываем «печатает...»
    typing_task = asyncio.create_task(send_typing(message))

    try:
        # Основная логика RAG
        BOX = answerGenerate(question)
        answer = BOX.answer
        answer = html.escape(answer)

        # Формируем красивый ответ
        response = f"✨ {answer}\n\n"

        # Логируем в базу (обрезаем длинный ответ для БД)
        SQL.stepDfInsert(
            name_sql_tab='auto_oracle_log',
            df=pd.DataFrame({
                'username': [f"@{username}"],
                'question': [question],
                'answer': [answer[:175]], 'delta_time':[BOX.delta_time_LLM],
                'search_type':[BOX.search_type]
            })
        )

        await typing_task  # дожидаемся завершения «печатает...»

        await message.answer(
            response,
            disable_web_page_preview=True
        )

    except Exception as e:
        await typing_task
        logger.error(f"Ошибка при обработке вопроса: {e}", exc_info=True)
        await message.answer(
            "Произошла ошибка при обработке запроса. Попробуйте позже или упростите вопрос."
        )


async def send_typing(message: Message):
    """Показывает индикатор «печатает…» пока генерируется ответ"""
    await bot.send_chat_action(message.chat.id, "typing")
    await asyncio.sleep(5)  # увеличил до 5 сек — у вас генерация ~1 минута


async def main():
    logger.info("Запуск бота автохимии (с прокси для РФ)...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, polling_timeout=30)


if __name__ == "__main__":
    asyncio.run(main())