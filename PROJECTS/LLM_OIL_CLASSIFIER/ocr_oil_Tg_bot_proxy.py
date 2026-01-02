# -*- coding: utf-8 -*-
# ocr_oil_Tg_bot_proxy.py
# Telegram-бот для RAG-системы по автохимии
# aiogram 3.x + llama.cpp + FAISS + прокси для России
#
# FIX: Telegram server says - Bad Request: message is too long
# -> отправляем ответ чанками (<= 4096 символов)

import os
import html
import asyncio
import logging
import platform
from typing import List

import pandas as pd

from dotenv import load_dotenv, find_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

# ← сессия с прокси
from aiogram.client.session.aiohttp import AiohttpSession

from bootstrap import *
from PROJECTS.LLM_OIL_CLASSIFIER.ocr_QA import answerGenerate


# ==========================
# CONFIG
# ==========================
TEST_mode = True
external_ai = True  # OpenRouter

# Telegram limit for a single message is 4096 characters
TG_LIMIT = 4096
SAFE_LIMIT = 3500  # запас под HTML/entities и непредвиденные расширения текста


# ==========================
# ENV / TOKEN
# ==========================
load_dotenv(find_dotenv())

if TEST_mode and (platform.system() == "Darwin"):
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # LavrGPT
    print(f"TEST_mode = {TEST_mode}")
else:
    BOT_TOKEN = os.getenv("TELEGRAM_MaxGPT_TOKEN")  # @Auto_Oracle_Bot


# ==========================
# PROXY
# ==========================
PROXY_URL = "http://M3MRa2:Q1Pp2Y@177.234.136.58:8000"  # ваш прокси
session = AiohttpSession(proxy=PROXY_URL)


# ==========================
# LOGGING
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("reinwell_ruseff_liquimoly")


# ==========================
# BOT / DISPATCHER
# ==========================
default_properties = DefaultBotProperties(parse_mode=ParseMode.HTML)
bot = Bot(token=BOT_TOKEN, session=session, default=default_properties)
dp = Dispatcher()


# ==========================
# HELPERS: long message split
# ==========================
def split_text_safely(text: str, limit: int = SAFE_LIMIT) -> List[str]:
    """
    Режет текст на чанки, стараясь резать по \n\n, затем по \n, затем по пробелу.
    """
    text = (text or "").strip()
    if not text:
        return []

    parts: List[str] = []
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        if cut < int(limit * 0.5):
            cut = text.rfind("\n", 0, limit)
        if cut < int(limit * 0.5):
            cut = text.rfind(" ", 0, limit)
        if cut < int(limit * 0.5):
            cut = limit  # fallback

        chunk = text[:cut].strip()
        if chunk:
            parts.append(chunk)
        text = text[cut:].strip()

    if text:
        parts.append(text)
    return parts


def close_unbalanced_b_tags(s: str) -> str:
    """
    Мини-страховка: если порезали и остался незакрытый <b>.
    (У тебя форматирование по правилам — в основном <b>...</b>)
    """
    opens = s.count("<b>")
    closes = s.count("</b>")
    if opens > closes:
        s += "</b>" * (opens - closes)
    return s


async def send_long_answer(message: Message, text: str) -> None:
    """
    Отправляет длинный текст несколькими сообщениями, чтобы не словить TG 'message is too long'.
    """
    chunks = split_text_safely(text, limit=SAFE_LIMIT)
    for ch in chunks:
        ch = close_unbalanced_b_tags(ch)

        # Последняя страховка на случай неожиданного раздувания строки
        if len(ch) > TG_LIMIT:
            ch = ch[: TG_LIMIT - 10] + "…"

        await message.answer(
            ch,
            disable_web_page_preview=True
        )


# ==========================
# HANDLERS
# ==========================
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
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"[{stamp}] Вопрос от @{username} (id={user_id}): {question}")

    # Показываем «печатает...»
    typing_task = asyncio.create_task(send_typing(message))

    try:
        BOX = answerGenerate(question, external_ai)
        answer = getattr(BOX, "answer", "") or ""

        # Экранируем под HTML-режим
        answer_safe = html.escape(answer)

        # Формируем ответ
        response = f"✨ {answer_safe}".strip()

        # Логируем в базу (обрезаем для БД)
        SQL.stepDfInsert(
            name_sql_tab='auto_oracle_log',
            df=pd.DataFrame({
                'username': [f"@{username}"],
                'question': [question],
                'answer': [answer_safe[:750]],
                'delta_time': [getattr(BOX, "delta_time_LLM", None)],
                'search_type': [getattr(BOX, "search_type", None)]
            })
        )

        await typing_task  # дожидаемся завершения «печатает...»

        # ВАЖНО: отправка чанками
        await send_long_answer(message, response)

    except Exception as e:
        await typing_task
        logger.error(f"Ошибка при обработке вопроса: {e}", exc_info=True)
        await message.answer(
            "Произошла ошибка при обработке запроса. Попробуйте позже или упростите вопрос."
        )


async def send_typing(message: Message):
    """Показывает индикатор «печатает…» пока генерируется ответ"""
    await bot.send_chat_action(message.chat.id, "typing")
    await asyncio.sleep(5)


async def main():
    logger.info("Запуск бота автохимии (с прокси для РФ)...")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, polling_timeout=30)


if __name__ == "__main__":
    asyncio.run(main())
