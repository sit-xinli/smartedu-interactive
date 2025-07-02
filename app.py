import gradio as gr
#import time
from openai import OpenAI
from pypdf import PdfReader
from moviepy import VideoFileClip
import dotenv
import tempfile
import os
import json
# Load environment variables from .env file
dotenv.load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

if OPENAI_API_KEY is None or OPENAI_BASE_URL is None :
    raise ValueError("Please set the OPENAI_API_KEY, OPENAI_URL environment variables.")
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def analyze_material(file, audio_prompt="以下の音声形式の教育資料をテキスト化してください", language="ja"):
    if file is None:
        yield "", "ファイルが選択されていません。"
        return

    yield "", "ファイル処理を開始します..."

    text = None
    if file.name.endswith(".pdf"):
        yield "", "PDFファイルを読み込んでいます..."
        pdf_reader = PdfReader(file.name)
        for page in pdf_reader.pages:
            if text is None:
                text = page.extract_text()
            else:              
                text += page.extract_text()
        yield "", "PDFファイルの読み込みが完了しました。"
    elif file.name.endswith(".mp4"):
        yield "", "MP4ファイルを処理しています..."
        video = VideoFileClip(file.name)
        audio_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        video.audio.write_audiofile(audio_path, logger=None)
        yield "", "音声抽出が完了しました。文字起こしを開始します..."
        with open(audio_path, "rb") as audio_file:
            transcript =client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe", 
                    file=audio_file,
                    prompt = audio_prompt, # Optional
                    language=language
                    )
            text = transcript.text
        os.remove(audio_path)
        yield "", "文字起こしが完了しました。"

    if text is not None:
        yield "", "OpenAIによる分析を開始します..."
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                        {   
                            "role": "system", 
                            "content": "あなたは、教材を分析し、詳細なダイジェストを提供するアシスタントです。"
                        },
                        {   
                            "role": "user", 
                            "content": f"以下の資料の詳細なダイジェストを提供する：\n\n{text}"
                        }
                    ],
            temperature=0.7,
            max_tokens=2000,
        )
        
        yield response.choices[0].message.content, "分析が完了しました。"
    
    else:
        yield "提供されたファイルからテキストを抽出できませんでした.", "エラー: テキスト抽出失敗。"

def interactive_qa(question, history, material_text):
    if not question:
        return "", history

    if not material_text:
        return "教材を分析してください。", history

    messages = []
    messages.append({"role": "system", "content": f"あなたは、以下の教材の内容に基づいて質問に答えるアシスタントです。\n\n教材:\n{material_text}"})

    for human_msg, ai_msg in history:
        messages.append({"role": "user", "content": human_msg})
        messages.append({"role": "assistant", "content": ai_msg})

    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        bot_reply = response.choices[0].message.content
    except Exception as e:
        bot_reply = f"エラーが発生しました: {e}"

    history.append([question, bot_reply])

    return "", history


def generate_quiz(material_text):
    if not material_text:
        return "教材を分析してください。", [], ""

    prompt = f"以下の教材に基づいて、多肢選択式のクイズ問題を1つ生成してください。質問、4つの選択肢、および正解をJSON形式で出力してください。\n\n教材:\n{material_text}\n\n例:\n{{\"question\": \"質問文\", \"options\": [\"選択肢A\", \"選択肢B\", \"選択肢C\", \"選択肢D\"], \"answer\": \"選択肢A\"}}"

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "あなたはクイズを生成するアシスタントです。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        quiz_json = response.choices[0].message.content
        quiz_data = json.loads(quiz_json)

        question = quiz_data.get("question", "")
        options = quiz_data.get("options", [])
        answer = quiz_data.get("answer", "")

        return question, gr.Radio(choices=options, value=None), answer

    except Exception as e:
        return f"クイズの生成中にエラーが発生しました: {e}", [], ""

def check_quiz_answer(question, selected_option, correct_answer):
    if selected_option == correct_answer:
        return f"正解です！ {correct_answer} が正解です。"
    else:
        return f"不正解です。あなたの選択は {selected_option} でした。正解は {correct_answer} です。"

with gr.Blocks() as demo:
    gr.Markdown("# スマート教育・プラットフォーム")

    with gr.Tabs():
        with gr.TabItem("Material Analysis"):
            with gr.Row():
                file_input = gr.File(label="Upload PDF or MP4")
                analyze_button = gr.Button("Analyze")
            analysis_output = gr.Markdown(label="Analysis Result")
            analysis_status = gr.Textbox(label="Status", interactive=False)

        with gr.TabItem("Interactive Q&A"):
            with gr.Row():
                question_input = gr.Textbox(label="Ask a question", scale=4)
                with gr.Column(scale=1):
                    qa_button = gr.Button("Submit", scale=1)
                    clear_button = gr.Button("Clear Chat")
            chatbot = gr.Chatbot()

        with gr.TabItem("Quiz"):
            with gr.Column():
                quiz_button = gr.Button("Generate Quiz")
                quiz_question = gr.Textbox(label="Question", interactive=False)
                quiz_options = gr.Radio(label="Options", interactive=True)
                check_answer_button = gr.Button("Check Answer")
            quiz_result = gr.Textbox(label="Result", interactive=False)
    quiz_answer = gr.State()

    # Event Handlers
    analyze_button.click(analyze_material, inputs=file_input, outputs=[analysis_output, analysis_status])
    qa_button.click(interactive_qa, inputs=[question_input, chatbot, analysis_output], outputs=[question_input, chatbot])
    clear_button.click(lambda: None, outputs=chatbot, queue=False)
    quiz_button.click(generate_quiz, inputs=analysis_output, outputs=[quiz_question, quiz_options, quiz_answer])
    check_answer_button.click(check_quiz_answer, inputs=[quiz_question, quiz_options, quiz_answer], outputs=quiz_result)

demo.launch()