# ocr_oil_Tg_bot.py
# Telegram-бот для RAG-системы по автохимии
# aiogram 3.x + llama.cpp + FAISS + прокси для России

import pandas as pd
import os

from bootstrap import *

import logging
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

class BoxSettings():

    def __int__(self):
        self.test_mode = False



        # определение типа устройтсва дл яразных ОС (Mac/Win) gpu/cpu


# Загружаем переменные окружения
load_dotenv(find_dotenv())
BOT_TOKEN = os.getenv("TELEGRAM_MaxGPT_TOKEN")

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
        "Задавай любой вопрос, например:\n\n"
        "• Как убрать нагар с клапанов без разборки двигателя?\n"
        "• Как разморозить замки автомобиля?\n"
        "• Чем удалить наклейку с кузова автомобиля?\n\n"
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
        answer = answerGenerate(question)
        answer = html.escape(answer)

        # Формируем красивый ответ
        response = f"✨ {answer}\n\n"

        # Логируем в базу (обрезаем длинный ответ для БД)
        SQL.stepDfInsert(
            name_sql_tab='auto_oracle_log',
            df=pd.DataFrame({
                'username': [f"@{username}"],
                'question': [question],
                'answer': [answer[:175]]
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