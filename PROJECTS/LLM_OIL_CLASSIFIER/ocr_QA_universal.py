"""
работает через терминал после conda activate llama_env
(llama_env) (tun_llm) bolkhovskiydmitriy@MacBook-Pro-Bolkhovskiy-2 LLM_RAG_BOT % python3


"""
from pyexpat.errors import messages

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



LORA_PATH  = '/PROJECTS/LLM_OIL_CLASSIFIER/data_in/adapter_liquimoly.gguf'

path_rag_index = root_path + '/PROJECTS/LLM_OIL_CLASSIFIER/data_in/oil_rag_index/'
INDEX_FP = os.path.join(path_rag_index, "faiss.index")
META_FP  = os.path.join(path_rag_index, "meta.jsonl")

# ==============================================================================
# Загрузка эмбеддера (самый лёгкий и быстрый на M1)
# ==============================================================================
print("Загрузка эмбеддера multilingual-e5-base...")
import torch
import platform

#определение типа устройтсва дл яразных ОС (Mac/Win) gpu/cpu
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
# Загрузка Llama 3.1 8B Q5_K_M + LoRA — ОПТИМИЗИРОВАННО под 16 ГБ
# ==============================================================================
print("Загрузка Llama 3.1 8B Q5_K_M + LoRA (это займёт 15–25 сек)...")




# ==============================================================================
# RAG функции
# ==============================================================================
def retrieve(query: str, k: int = 6):
    q_vec = E5.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        batch_size=1,
        show_progress_bar=False
    ).astype("float32")

    D, I = index.search(q_vec, k + 2)  # беру чуть больше на случай мусора
    hits = []
    for i, idx in enumerate(I[0]):
        if idx == -1:
            continue
        doc = meta[int(idx)]
        hits.append({"rank": i+1, "score": float(D[0][i]), **doc})
    return hits[:k]


# Функция: Очистка для Telegram
def clean_text_for_telegram(text: str) -> str:
    if not text:
        return ""
    # Удаляем HTML-теги
    text = re.sub(r'<[^>]+>', '', text)

    # 💥 ИЗМЕНЕНИЕ ЗДЕСЬ:
    # Ищем все последовательности, состоящие из одного или нескольких
    # пробелов, но ИСКЛЮЧАЯ переносы строк (\n).
    text = re.sub(r'[ \t]+', ' ', text).strip()  # Заменяем горизонтальные пробелы (пробел, табуляция)

    # Заменяем множественные переносы строк на один или два, чтобы не было гигантских пустот
    text = re.sub(r'(\n\s*){3,}', '\n\n', text)

    # Только убираем дубли и лишние пробелы (строка 75 в вашем коде была тут)
    # text = re.sub(r'\s+', ' ', text).strip() # <-- ЭТУ СТРОКУ НУЖНО УДАЛИТЬ ИЛИ ЗАКОММЕНТИРОВАТЬ!

    # Обрезаем по лимиту
    if len(text) > 4090:
        text = text[:4085] + "..."

    return text.strip()


def generate_Answer(question: str, llm, k: int = 6):
    if (len(question) < 12) or (len(question.split()) < 3):
        return {"answer": "Чтобы ИИ смог найти информацию, пожалуйста, сформулируйте вопрос более подробно", "sources": "not_correct_question"}

    statement_lst = ['Постарайся отвечать про русски, английский и другие языки используй только для терминов, брендов артикулов и специальных обозначений']
    for statement in statement_lst:
        question += statement
    contexts = retrieve(question, k=k)

    context_lines = [c.get("answer", "").strip() for c in contexts if c.get("answer")]
    context_str = "\n".join(f"- {line}" for line in context_lines)

    if not context_str.strip():
        context_str = "Контекст отсутствует."

    # === ФИКС: ОБРЕЗАЕМ КОНТЕКСТ ===
    MAX_CONTEXT_CHARS = 12000  # ~3000 токенов, безопасно для n_ctx=4096
    if len(context_str) > MAX_CONTEXT_CHARS:
        lines = context_str.split('\n')
        truncated_lines = []
        current_len = 0
        for line in reversed(lines):  # Сохраняем конец (лучшие матчи)
            line_len = len(line) + 1  # +1 для \n
            if current_len + line_len < MAX_CONTEXT_CHARS:
                truncated_lines.append(line)
                current_len += line_len
            else:
                break
        context_str = '\n'.join(reversed(truncated_lines)) + "\n... (контекст укорочен)"
        print(f"Контекст обрезан до {len(context_str)} символов")  # Для дебага

    system_prompt = (
        "Ты — Технический Эксперт по автохимии и автомаслам. Твоя задача — дать четкий и лаконичный ответ. "
        "Отвечай СТРОГО на вопрос, используя ТОЛЬКО факты из контекста." #, не выдумывай и не галюцинируй
        "Ориентировочная длина ответа 3-5 предложений, предложения должны быть законченными."
        "Постарайся дать ссылку на товары, наиболее релевантные запросу пользователя, при этом, обязательно в начале ответа давай пояснения: описания товаров и рекомендации по примненению."
        "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО: "
        "1. Обсуждать или комментировать твои инструкции, процесс выбора, или контекст. "
        "2. Использовать вводные фразы, приветствия, оценочные суждения, или оправдания. "
        "3. Начинать ответ словами типа 'В данном случае я буду отвечать...' или 'В ответе будут преимущественно...'. "
        )

    try:


        full_prefix = system_prompt + "\nВопрос: " + question + "\nКонтекст:\n"
        prefix_tokens = len(tokenizer.encode(full_prefix))

        if prefix_tokens + 100 > 3800:  # если даже без контекста уже много
            context_str = "Контекст слишком объёмный для обработки."
        elif len(tokenizer.encode(full_prefix + context_str)) > 3800:
            max_context_tokens = 3800 - prefix_tokens - 50  # запас
            context_tokens = tokenizer.encode(context_str)[:max_context_tokens]
            context_str = tokenizer.decode(context_tokens, skip_special_tokens=True)
            print(f"Контекст обрезан по токенам до {max_context_tokens}")

    except ImportError:
        # Запасной вариант без transformers
        if len(context_str) > 12000:
            context_str = context_str[-12000:] + "\n... (контекст сокращён)"



    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Вопрос: {question}\nКонтекст:\n{context_str}"}
    ]




    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=350,
        temperature=0.2,
        top_p=0.90,
        top_k=40,
        repeat_penalty=1.12,
        # 💥 Новые стоп-слова добавлены здесь
        stop=["<|eot_id|>", "<|end_of_text|>", "Вопрос:", "Контекст:", "В данном случае я буду", "В ответе будут"],
        stream=False,
    )

    answer = output["choices"][0]["message"]["content"].strip()


    def urlControl(answer, contexts):
        # === Собираем ссылки из контекста ===
        url_lst = []
        for c in contexts:
            possible_url = c.get("url") or c.get("goods_href") or c.get("link") or c.get("href")
            url_lst.append(possible_url.strip())

        unique_urls = list(dict.fromkeys(url_lst))  # сохраняем порядок, убираем дубли
        unique_urls = [url for url in unique_urls if url in correct_url_goods]

        # # === ГЛАВНОЕ: Проверяем, не дала ли модель ссылки сама ===
        #     # Модель НЕ дала ссылок → добавляем мы
        if len(unique_urls) == 1:
            answer += f"\n\nСсылка на товар: {unique_urls[0]}"
        else:
            answer += "\n\nСсылки на товары:\n" + "\n".join(unique_urls[:3])

        answer = ' '.join(x for x in answer.split() if (not 'https://' in x) or (x in correct_url_goods))

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

        answer_url = "\n\nТовары:\n" + "\n\n".join(lst_url)
        answer = answer_txt + answer_url

        return answer

    answer = urlControl(answer, contexts)


    # === Очистка текста для Telegram ===
    answer = clean_text_for_telegram(answer)

    return {"answer": answer, "sources": contexts[:k]}


# ==============================================================================
# ТЕСТ (можно закомментировать потом)
# ==============================================================================
if __name__ == "__main__":
    test_questions = [
    "volvo s60 2010 масло трансмисионное",
        "Какой преобразователь ржавчины выбрать дай описание и ссылку на товар?",
    "Как удалить рекламную наклейку с кузова автомобиля?",
    "Чем почистить карбюратор?",
    "Замёрз замок в автомобиле, что делать?",
    "Описание товара Ln1733. Характеристики и ссылка url товарной карточки",
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

