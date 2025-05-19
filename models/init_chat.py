from fastapi import HTTPException

from core.firebase import db, bucket
from core.gemini import model, multimodal_model
from utils.semantic_search import find_relevant_chunks, extract_pdf_text_and_tables_from_blob
from utils.makroeconomics import (
    two_wheels_model, four_wheels_model, retail_general_model,
    retail_beauty_model, retail_fnb_model, retail_drugstore_model
)
from utils.search_web import search_web_snippets
from utils.follow_up_question import (
    recommend_follow_up_questions_gm,
    recommend_follow_up_questions_ngm
)
from PIL import Image
from datetime import datetime

import io

class Chat:
    def __init__(self):
        self.now = datetime.now()

    async def generate_response(self, chat_option, prompt, file_id_input):
        file_url = None
        references = None

        if chat_option == "General Macroeconomics":
            if file_id_input:
                file_url, response = self._handle_file_prompt(prompt, file_id_input)
                follow_up_question = recommend_follow_up_questions_gm(prompt, response, file_id_input)
            else:
                response, references = self._handle_web_prompt(prompt)
                follow_up_question = recommend_follow_up_questions_gm(prompt, response)
        else:
            response = self._handle_custom_model(chat_option, prompt)
            follow_up_question = recommend_follow_up_questions_ngm(prompt, response, chat_option)
            file_id_input = None

        return response, file_url, references, follow_up_question

    def _handle_file_prompt(self, prompt, file_id_input):
        file_doc = db.collection('files').document(file_id_input).get()
        if not file_doc.exists:
            raise HTTPException(status_code=404, detail="File not found")

        file_data = file_doc.to_dict()
        file_url = file_data.get('url')
        content_type = file_data.get('content_type')
        storage_path = file_data.get('storage_path')

        blob = bucket.blob(storage_path)
        file_content = blob.download_as_bytes()

        if 'application/pdf' in content_type:
            pdf_text = extract_pdf_text_and_tables_from_blob(file_content)
            relevant_text = find_relevant_chunks(pdf_text, prompt, chunk_size=650, top_k=5) if len(pdf_text) > 1000 else pdf_text

            response = model.generate_content(
                f"""
                    Kamu adalah **SPLASHBot**, AI Agent yang mengkhususkan diri dalam **analisis dokumen ekonomi**, khususnya file **PDF** yang diberikan oleh pengguna.

                    ### Informasi yang Disediakan:
                    - **Konten relevan dari PDF**:  
                    {relevant_text}

                    - **Pertanyaan dari pengguna**:  
                    "{prompt}"

                    ### Aturan Penting:
                    1. **Hanya jawab pertanyaan** jika isi PDF berkaitan dengan **ekonomi**.  
                    Jika tidak relevan secara ekonomi, jawab dengan:  
                    _"Maaf, saya hanya dapat menjawab pertanyaan yang berkaitan dengan ekonomi."_
                    2. Soroti **kata kunci penting** dalam jawaban dengan **bold** agar mudah dikenali.
                    3. Jawaban harus **jelas**, **fokus pada konteks ekonomi**, dan **berdasarkan isi PDF**.

                    ### Tugas:
                    Berikan jawaban berbasis analisis isi PDF tersebut, dengan tetap menjaga fokus pada aspek ekonomi dan pertanyaan pengguna.
                """
            ).text

        elif content_type.startswith('image/'):
            image = Image.open(io.BytesIO(file_content)).convert("RGB")
            response = multimodal_model.generate_content(
               [
                    "Kamu adalah **SPLASHBot**, AI analis yang **mengkhususkan diri di bidang ekonomi dan bisnis**, serta mampu **menganalisis gambar** yang relevan dengan topik tersebut.",
                    image,
                    f"""
                        ### Konteks:
                        - Pertanyaan dari pengguna: **{prompt}**

                        ### Instruksi:

                        1. Tinjau gambar yang diberikan.
                        2. Jika **gambar tidak berkaitan dengan topik ekonomi atau bisnis**, **jangan memberikan jawaban apapun** selain menyatakan bahwa gambar tidak relevan.
                        3. Jika gambar relevan, berikan **analisis ekonomi atau bisnis yang tajam dan bernilai**.
                        4. Soroti **kata kunci penting** dalam jawaban dengan format **bold** untuk penekanan.
                        5. Jawaban harus **padat, profesional, dan bernilai insight**—hindari narasi yang terlalu panjang atau di luar topik.

                        ### Tujuan:
                        Memberikan analisis **berbasis visual** dengan fokus pada **makna ekonomi**, seperti tren pasar, perilaku konsumen, pertumbuhan, distribusi wilayah, dsb.
                    """
                ]
            ).text

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        return file_url, response

    def _handle_web_prompt(self, prompt):
        result = search_web_snippets(prompt, num_results=5)
        references = result.get("linked_results", [])
        snippets = result.get("snippet_results", "")

        response = model.generate_content(
            f"""
            Kamu adalah **SPLASHBot**, sebuah AI Agent yang ahli dalam menjawab pertanyaan seputar **ekonomi**, termasuk ekonomi makro, mikro, kebijakan fiskal/moneter, perdagangan, keuangan, dan indikator ekonomi.

            ### Pertanyaan dari Pengguna:
            {prompt}

            ### Informasi Terkini dari Internet:
            {snippets}

            ### Referensi:
            {references}

            ### Catatan Penting:
            - Gunakan **informasi dari internet** dan semua pengetahuan-mu jika **relevan dengan topik ekonomi**.
            - **Abaikan** informasi yang tidak berkaitan dengan ekonomi.
            - **Jangan menyebutkan atau mengutip link** dari internet secara eksplisit dalam jawaban.
            - Gunakan penekanan (**bold**) pada **kata kunci penting** dalam jawaban agar lebih jelas bagi pengguna.
            - Sebisa mungkin, jangan menjawab dengan "saya tidak tahu" atau "saya tidak bisa menjawab". Gunakan pengetahuan yang ada untuk memberikan jawaban yang informatif.

            ### Tugasmu:
            Berikan jawaban yang **jelas**, **relevan**, dan **berbasis ekonomi** terhadap pertanyaan pengguna. 
            Jika pertanyaannya **tidak berkaitan dengan ekonomi**, cukup balas dengan: _"Maaf, saya hanya dapat menjawab pertanyaan yang berkaitan dengan ekonomi."_
            """
        ).text

        return response, references

    def _handle_custom_model(self, chat_option, prompt):
        model_map = {
            "2 Wheels": two_wheels_model,
            "4 Wheels": four_wheels_model,
            "Retail General": retail_general_model,
            "Retail Beauty": retail_beauty_model,
            "Retail FnB": retail_fnb_model,
            "Retail Drugstore": retail_drugstore_model
        }

        model_func = model_map.get(chat_option)
        if not model_func:
            raise HTTPException(status_code=400, detail="Invalid chat option")
        
        return model_func(prompt)