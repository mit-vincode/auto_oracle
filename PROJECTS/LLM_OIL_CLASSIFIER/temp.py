"""
работает через терминал после conda activate llama_env
(llama_env) (tun_llm) bolkhovskiydmitriy@MacBook-Pro-Bolkhovskiy-2 LLM_RAG_BOT % python3

УЛУЧШЕННАЯ ВЕРСИЯ с оптимизированным промптингом
"""

# -*- coding: utf-8 -*-
# 100% рабочая версия под MacBook Pro M1/M2 16 ГБ (ноябрь 2025)

from bootstrap import *

correct_url_goods = (U24.data2Df_upload(root_path + '/PROJECTS/LLM_OIL_CLASSIFIER/data_in/alleya_lavr_to_json.xlsx')[
                         'url'].unique())

import os
import json, re
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from datetime import datetime as DT
from aiogram.utils.markdown import hlink

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", use_fast=True)

LORA_PATH = '/PROJECTS/LLM_OIL_CLASSIFIER/data_in/adapter_liquimoly.gguf'

path_rag_index = root_path + '/PROJECTS/LLM_OIL_CLASSIFIER/data_in/oil_rag_index/'
INDEX_FP = os.path.join(path_rag_index, "faiss.index")
META_FP = os.path.join(path_rag_index, "meta.jsonl")

# ==============================================================================
# Загрузка эмбеддера (самый лёгкий и быстрый на M1)
# ==============================================================================
print("Загрузка эмбеддера multilingual-e5-base...")
import torch
import platform

# определение типа устройтсва дл яразных ОС (Mac/Win) gpu/cpu
if platform.system() != "Darwin":  # если не macOS
    mps_availability = False
else:
    mps_availability = torch.backends.mps.is_available()

if torch.cuda.is_available():
    device = "cuda"
elif mps_availability:
    device = "mps"
else:
    device = "cpu"

print(f"[E5] Устройство: {device} (OS: {platform.system()})")

E5 = SentenceTransformer(
    "intfloat/multilingual-e5-base",
    device=device,
    cache_folder="./models_cache"
)

# ==============================================================================
# Загрузка FAISS индекса
# ==============================================================================
print("Загрузка FAISS индекса...")
index = faiss.read_index(INDEX_FP)
with open(META_FP, "r", encoding="utf-8") as f:
    meta = [json.loads(l) for l in f if l.strip()]

# ==============================================================================
# УЛУЧШЕННЫЙ СИСТЕМНЫЙ ПРОМПТ (на основе успешного паттерна из ocr_QA.py)
# ==============================================================================
SYSTEM_PROMPT_V2 = (
    "Ты — технический эксперт по автохимии и автомаслам. "
    "Отвечай ТОЛЬКО полезной информацией из КОНТЕКСТА. "
    "НИКАКИХ заголовков, подзаголовков, вступлений, размышлений о процессе, пустых строк.\n\n"

    "КРИТИЧЕСКИЕ ПРАВИЛА:\n"
    "1. Отвечай СРАЗУ с фактов, без вводных слов.\n"
    "2. Используй только жирный текст (**) для названий продуктов, обычный текст для описаний, • для пунктов.\n"
    "3. Если данных нет — пиши: «В базе нет информации по этому вопросу».\n"
    "4. НИКОГДА не пиши: «В данном случае», «Я буду отвечать», «Вот информация», «Рекомендую», «Заголовок», «Ответ».\n"
    "5. Запрещено начинать ответ с обсуждения контекста или инструкций.\n"
    "6. Ссылки добавляются автоматически — НЕ выдумывай URL.\n\n"

    "ПРИМЕР ПРАВИЛЬНОГО ОТВЕТА:\n"
    "**LIQUI MOLY Pro-Line Jetclean Benzin** — концентрированный очиститель инжектора.\n"
    "• Применение: добавить 500 мл на 70 л топлива\n"
    "• Эффект: удаляет нагар, восстанавливает распыл\n"
    "• Периодичность: каждые 5000 км\n\n"

    "Для старых моторов с карбюратором используйте **LIQUI MOLY Vergaser-Aussen-Reiniger** — спрей для внешней очистки."
)

# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ: Предобработка вопроса
# ==============================================================================
BOILERPLATE_PREFIXES = [
    'доброго всем здравия',
    'начну с самого начала',
    'подскажите пожалуста',
    'подскажите', 'пожалуйста',
    'нужен совет',
    'спасибо за ответ',
    'вопрос по',
    'несколько вопросов',
    'здравствуйте',
    'скажите',
    'помогите',
    'таки говоря',
    'возник вопрос',
    'и снова',
    'добрый день',
    'доброго времени суток',
    'добрый вечер',
    'доброго дня', 'день добрый', 'доброго времени', 'спасибо'
]

_boiler_re_parts = [rf'(?:^|(?<=[\.\!\?]\s))\s*{re.escape(p)}\s*[,:-]?\s*' for p in BOILERPLATE_PREFIXES]
BOILER_RE = re.compile('|'.join(_boiler_re_parts), re.IGNORECASE | re.UNICODE)


def _strip_boilerplate(s: str) -> str:
    """Удаление шаблонных фраз из вопроса"""
    if not isinstance(s, str):
        return ''
    prev, cur = None, s
    while prev != cur:
        prev = cur
        cur = BOILER_RE.sub('', cur).strip()
    return re.sub(r'\s+', ' ', cur, flags=re.UNICODE).strip()


# ==============================================================================
# УЛУЧШЕННАЯ ФУНКЦИЯ RETRIEVE с дедупликацией
# ==============================================================================
def retrieve(query: str, k: int = 6):
    """
    Улучшенная версия retrieve с:
    - Дедупликацией по содержимому
    - Фильтрацией пустых результатов
    - Сортировкой по релевантности
    """
    q_vec = E5.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        batch_size=1,
        show_progress_bar=False
    ).astype("float32")

    # Берём больше результатов для фильтрации
    D, I = index.search(q_vec, k * 2)

    hits = []
    seen_content = set()  # Дедупликация

    for i, idx in enumerate(I[0]):
        if idx == -1:
            continue

        doc = meta[int(idx)]
        content = doc.get("answer", "").strip()

        # Пропускаем пустые или дубликаты
        if not content or content in seen_content:
            continue

        # Фильтруем слишком короткие (менее 50 символов)
        if len(content) < 50:
            continue

        seen_content.add(content)
        hits.append({
            "rank": len(hits) + 1,
            "score": float(D[0][i]),
            **doc
        })

        if len(hits) >= k:
            break

    return hits


# ==============================================================================
# УМНАЯ ОБРЕЗКА КОНТЕКСТА с приоритизацией
# ==============================================================================
def truncate_context_smart(contexts: list, max_tokens: int = 3000):
    """
    Умная обрезка контекста:
    1. Приоритет лучшим матчам (низкий score)
    2. Сохранение целостности предложений
    3. Индикаторы обрезки
    """
    # Сортируем по релевантности
    sorted_contexts = sorted(contexts, key=lambda x: x.get("score", 999))

    context_parts = []
    current_tokens = 0

    for ctx in sorted_contexts:
        content = ctx.get("answer", "").strip()
        if not content:
            continue

        # Приблизительная оценка токенов (1 токен ≈ 4 символа для русского)
        estimated_tokens = len(content) // 4

        if current_tokens + estimated_tokens > max_tokens:
            # Если не влезает, обрезаем по предложениям
            remaining_tokens = max_tokens - current_tokens
            remaining_chars = remaining_tokens * 4

            # Берём полные предложения
            sentences = re.split(r'(?<=[.!?])\s+', content)
            truncated = []
            temp_len = 0

            for sent in sentences:
                if temp_len + len(sent) < remaining_chars:
                    truncated.append(sent)
                    temp_len += len(sent)
                else:
                    break

            if truncated:
                context_parts.append(" ".join(truncated))
            break

        context_parts.append(f"- {content}")
        current_tokens += estimated_tokens

    result = "\n".join(context_parts)

    if current_tokens >= max_tokens:
        result += "\n... (контекст сокращён до наиболее релевантного)"

    return result


# ==============================================================================
# КОНСТРУИРОВАНИЕ ОПТИМИЗИРОВАННОГО ЗАПРОСА
# ==============================================================================
def construct_optimized_query(question: str, context_str: str) -> list:
    """
    Конструирует запрос по принципам успешного промпта из ocr_QA.py

    Ключевые улучшения:
    - Чёткая структура с разделителями <<<>>>
    - Явное указание задачи в конце
    - Запрет на meta-комментарии
    """
    # Очистка вопроса
    question_clean = question.strip()
    if not question_clean.endswith('?'):
        question_clean += '?'

    # Формирование запроса с чёткой структурой
    user_content = (
        f"<<<ВОПРОС КЛИЕНТА>>> {question_clean} <<</ВОПРОС КЛИЕНТА>>>\n\n"
        f"<<<ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ>>>\n{context_str}\n<<</ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ>>>\n\n"
        f"**Задача:** Дай лаконичный структурированный ответ (3-5 предложений). "
        f"Используй ТОЛЬКО факты из технической информации выше."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_V2},
        {"role": "user", "content": user_content}
    ]

    return messages


# ==============================================================================
# Функция: Очистка для Telegram
# ==============================================================================
def clean_text_for_telegram(text: str) -> str:
    if not text:
        return ""
    # Удаляем HTML-теги
    text = re.sub(r'<[^>]+>', '', text)

    # Заменяем горизонтальные пробелы (но не \n)
    text = re.sub(r'[ \t]+', ' ', text).strip()

    # Заменяем множественные переносы строк
    text = re.sub(r'(\n\s*){3,}', '\n\n', text)

    # Удаляем артефакты промпта
    text = re.sub(r'^(Ответ:|Заголовок:|##?\s*)', '', text, flags=re.MULTILINE)

    # Обрезаем по лимиту
    if len(text) > 4090:
        text = text[:4085] + "..."

    return text.strip()


# ==============================================================================
# КОНТРОЛЬ И ДОБАВЛЕНИЕ URL
# ==============================================================================
def urlControl(answer, contexts):
    """Добавление ссылок на товары из контекста"""
    # Собираем ссылки из контекста
    url_lst = []
    for c in contexts:
        possible_url = c.get("url") or c.get("goods_href") or c.get("link") or c.get("href")
        if possible_url:
            url_lst.append(possible_url.strip())

    unique_urls = list(dict.fromkeys(url_lst))  # сохраняем порядок, убираем дубли
    unique_urls = [url for url in unique_urls if url in correct_url_goods]

    # Добавляем ссылки
    if len(unique_urls) == 1:
        answer += f"\n\nСсылка на товар: {unique_urls[0]}"
    elif len(unique_urls) > 1:
        answer += "\n\nСсылки на товары:\n" + "\n".join(unique_urls[:3])

    # Удаляем выдуманные URL из ответа модели
    answer = ' '.join(x for x in answer.split() if (not 'https://' in x) or (x in correct_url_goods))

    # Разделяем текст и ссылки
    answer = answer.replace("Ссылка на", '|').replace("Ссылки на", '|').split("|")
    answer = [x for x in answer if (len(x) > 25) or ('https://' in x)]
    answer_txt = [x for x in answer if 'https://' not in x]
    answer_url = [x for x in answer if 'https://' in x]

    answer_txt = '\n'.join(answer_txt)
    lst_url = []
    for x in answer_url:
        x = x.split()
        for y in x:
            if 'https://' in y:
                lst_url.append(y)

    answer_url = "\n\nТовары:\n" + "\n\n".join(lst_url) if lst_url else ""
    answer = answer_txt + answer_url

    return answer


# ==============================================================================
# ДИНАМИЧЕСКОЕ ОПРЕДЕЛЕНИЕ max_tokens
# ==============================================================================
def calculate_max_tokens(num_contexts: int, question_length: int) -> int:
    """
    Динамическое определение max_tokens на основе сложности запроса
    """
    base_tokens = 250

    # Чем больше контекста, тем больше нужно токенов
    context_bonus = min(num_contexts * 30, 150)

    # Длинные вопросы требуют развёрнутых ответов
    question_bonus = min(question_length // 20, 50)

    total = base_tokens + context_bonus + question_bonus

    # Ограничение сверху
    return min(total, 500)


# ==============================================================================
# ГЛАВНАЯ ФУНКЦИЯ: УЛУЧШЕННАЯ generate_Answer
# ==============================================================================
def generate_Answer(question: str, llm, k: int = 6):
    """
    УЛУЧШЕННАЯ ВЕРСИЯ с оптимизированным промптингом

    Ключевые улучшения:
    1. Предобработка вопроса от шаблонных фраз
    2. Дедупликация результатов поиска
    3. Умная обрезка контекста с приоритизацией
    4. Структурированный промпт с разделителями <<<>>>
    5. Расширенные stop-слова
    6. Динамический max_tokens
    """

    # Предобработка вопроса
    question = _strip_boilerplate(question)

    # Валидация
    if (len(question) < 12) or (len(question.split()) < 3):
        return {
            "answer": "Для точного ответа сформулируйте вопрос подробнее (минимум 3 слова).",
            "sources": "not_correct_question"
        }

    # Retrieve с дедупликацией
    contexts = retrieve(question, k=k)

    if not contexts:
        return {
            "answer": "В базе нет информации по этому вопросу. Попробуйте переформулировать запрос.",
            "sources": []
        }

    # Умная обрезка контекста
    context_str = truncate_context_smart(contexts, max_tokens=3000)

    if not context_str.strip():
        context_str = "Релевантная информация не найдена."

    # Дополнительная токеновая проверка
    try:
        full_prefix = SYSTEM_PROMPT_V2 + "\nВопрос: " + question + "\nКонтекст:\n"
        prefix_tokens = len(tokenizer.encode(full_prefix))

        if prefix_tokens + 100 > 3800:
            context_str = "Контекст слишком объёмный для обработки."
        elif len(tokenizer.encode(full_prefix + context_str)) > 3800:
            max_context_tokens = 3800 - prefix_tokens - 50
            context_tokens = tokenizer.encode(context_str)[:max_context_tokens]
            context_str = tokenizer.decode(context_tokens, skip_special_tokens=True)
            print(f"Контекст обрезан по токенам до {max_context_tokens}")
    except Exception as e:
        print(f"Ошибка токенизации: {e}")
        # Запасной вариант
        if len(context_str) > 12000:
            context_str = context_str[-12000:] + "\n... (контекст сокращён)"

    # Конструирование оптимизированного запроса
    messages = construct_optimized_query(question, context_str)

    # РАСШИРЕННЫЙ список stop-слов (критическое улучшение!)
    STOP_WORDS = [
        "<|im_end|>",
        "<|eot_id|>",
        "<|end_of_text|>",
        "```",
        "Вопрос:",
        "Контекст:",
        "ВОПРОС КЛИЕНТА",
        "ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ",
        "В данном случае я",
        "Я буду отвечать",
        "Вот информация",
        "Заголовок:",
        "Ответ:",
        "Рекомендую",
        "\n\n\n\n",  # 4+ переноса строки
    ]

    # Динамический расчёт max_tokens
    max_tokens = calculate_max_tokens(len(contexts), len(question))

    # Генерация с оптимальными параметрами
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,  # Как в успешном скрипте
        top_p=0.90,
        top_k=40,
        repeat_penalty=1.15,  # Чуть выше для борьбы с повторами
        stop=STOP_WORDS,
        stream=False,
    )

    answer = output["choices"][0]["message"]["content"].strip()

    # Пост-обработка: удаление артефактов
    answer = re.sub(r'^(Ответ:|Заголовок:|##?\s*)', '', answer, flags=re.MULTILINE)
    answer = re.sub(r'\n{3,}', '\n\n', answer)

    # Контроль URL
    answer = urlControl(answer, contexts)

    # Очистка для Telegram
    answer = clean_text_for_telegram(answer)

    return {"answer": answer, "sources": contexts[:k]}


if __name__ == "__main__":
    test_questions = ["как разморозить замки автомобиля?",
                      'может ли антигель навредить дизельному двигателю?',
    " Как работают присадки для улучшения работы гидрокомпенсаторов",
        "Какой преобразователь ржавчины выбрать?",
    "Как удалить  наклейку с кузова автомобиля?",
    "Чем почистить карбюратор?",
    "Чем почистить пятно от чая на диван?",
    "Нужны ли антикоррозийные присадки для защиты топливного бака?",
    "Как работают присадки для улучшения работы гидрокомпенсаторов?",
    "Сколько раз в год можно добавлять присадки в трансмиссионное масло АКПП?",
    "Эффективны ли депрессорные присадки против замерзания дизельного топлива зимой?",
    "Как промыть радиатор системы охлаждения с помощью специальной присадки?",
    "Какие очистители карбюратора стоит использовать для старых инжекторных моторов?",
    "Присадки для цепи ГРМ: польза и возможный вред?",
    "Как выбрать универсальную присадку для смешанного топлива (бензин + этанол)?"
]


    ##################
    from llama_cpp import Llama

    model_dir = root_path + '/DATA_CATALOGS/llm_models/gguf/'

    # MODEL_PATH = model_dir + '//Qwen2.5-Coder-3B-Instruct.Q5_K_M.gguf' #гораздо хуже Mistral
    # MODEL_PATH = model_dir + '/DeepSeek-R1-Distill-Qwen-7B.Q4_K_M.gguf' #гораздо хуже Mistral
    ## Полный отстой: Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf, Phi-3-mini-4k-instruct (Microsoft)

    MODEL_PATH = model_dir + '/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf'  # наилучший вариант по качеству и скорости

    MAX_len_param_df = 25  # в ячейке A_text в среднем 400 символов - кол-во строк больше, значит срез не качественный - общий поиск
    LIM_len_param_df = 13

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=8192,  # 8192
        n_batch=1024,
        n_gpu_layers=99,
        n_threads=10,
        verbose=False,
    )

    ##################

    T1 = U24.tNow()
    for i, query in enumerate(test_questions):
        tt1 = U24.tNow()
        result = generate_Answer(query, llm, k=4)
        answer = result["answer"]
        print(f"\ntime_delta = {(U24.tNow() - tt1).total_seconds()}\nзапрос: {query}\nответ: {answer}\n{'=' * 60}")

    print(f"~ среднее время на 1 запрос: {((U24.tNow() - T1).total_seconds()) // (i + 1)}")

