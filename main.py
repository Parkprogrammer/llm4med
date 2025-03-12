from flask import Flask, request, jsonify, render_template_string, send_from_directory
from vllm import LLM, SamplingParams
# import torch
import pandas as pd
import numpy as np
from pyngrok import ngrok
import os
import json
from datetime import datetime
import traceback
import re
from collections import Counter
from pathlib import Path

from Recsys import build_tfidf_index, extract_keywords_from_output, find_best_matches_tfidf

from template import STAGE_1_TEMPLATE, STAGE_2_TEMPLATE, STAGE_3_TEMPLATE
from template import HTML_TEMPLATE # flask를 활용한 프로토타입 랜더링 front-end

from model import safe_get, LangchainCOTProcessor

from utils import setup_ngrok, load_ngrok_config, close_ngrok
from utils import translate_to_korean


# 전역 변수로 현재 처리 중인 데이터 상태 관리
current_data = {
    'df': None,
    'current_row': 0,
    'total_rows': 0
}

processor = LangchainCOTProcessor("TsinghuaC3I/Llama-3-8B-UltraMedical")

# TODO: 콘텐츠 이미지가 담긴 디렉토리로 경로 직접 지정 ex. /content/drive/MyDrive/Colab Notebooks/llm4med/Graph Neural Network/content
IMAGE_DIRECTORY = None
# TODO: 전처리한 데이터를 입력해주는 경로 ex."/content/drive/MyDrive/Colab Notebooks/llm4med/Graph Neural Network/recommendation.csv"
RECOMMENDATION_PATH = None


app = Flask(__name__)

# 전역 변수
tfidf_vectorizer = None
tfidf_matrix = None
global_content_list = []

def load_content_csv(csv_path):
    df = pd.read_csv(csv_path, encoding='cp949')
    content_list = []
    for _, row in df.iterrows():
        # 전체 키워드 컬럼 예: "당뇨, 당뇨병, 당화혈색소, ..."
        # 이를 쉼표로 split하여 strip
        all_keywords = []
        if pd.notna(row["Keywords"]):
            all_keywords = [k.strip() for k in row["Keywords"].split(',')]

        content_list.append({
            "No": row["No"],
            "Title": row["Title"],
            "File_name": row["File_name"],
            "Keywords": all_keywords
        })
    return content_list


@app.route('/image/<path:filename>')
def serve_image(filename):
    """이미지 파일 서빙"""
    try:
        # 확장자가 없는 경우 .png 추가
        if not filename.lower().endswith('.png'):
            filename = f"{filename}.png"

        print(f"Serving image: {filename}")
        return send_from_directory(IMAGE_DIRECTORY, filename)
    except Exception as e:
        print(f"Error serving image: {e}")
        return '', 404

@app.errorhandler(Exception)
def handle_exception(e):
    # 자세한 에러 트레이스백 출력
    print("=== Error Traceback ===")
    traceback.print_exc()
    print("=====================")
    return jsonify({
        'error': str(e),
        'traceback': traceback.format_exc()
    }), 500

@app.route('/patients_list', methods=['GET'])
def get_patients_list():
    """
    쿼리스트링 page=? 로 페이지 번호를 받고,
    그 페이지에 해당하는 10개 행을 JSON 형식으로 반환
    예: /patients_list?page=2
    """
    try:
        if current_data['df'] is None:
            return jsonify({
                'patients': [],
                'page': 1,
                'total_pages': 1
            }), 200

        # page 파라미터 읽기 (기본 1)
        page_str = request.args.get('page', '1')
        page = int(page_str)
        per_page = 10  # 페이지당 10개

        df = current_data['df']
        total_rows = len(df)
        total_pages = (total_rows + per_page - 1) // per_page  # 올림

        # 페이지 범위 초과하면 빈 리스트
        if page < 1 or page > total_pages:
            return jsonify({
                'patients': [],
                'page': page,
                'total_pages': total_pages
            })

        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_df = df.iloc[start_idx:end_idx]

        # 반환할 필드들만 골라서 JSON 만들기
        # 예: "No", "Disease Classification", "Gender", "Age", "BMI" 등 간략화된 정보
        patients_data = []
        for i, row in page_df.iterrows():
            item = {
                'index': int(i),  # 실제 행 인덱스
                'disease_classification': f"{safe_get(row, 'Disease Classification - Primary')} / {safe_get(row, 'Disease Classification - Secondary')}",
                'departments': f"{safe_get(row, 'Department - Main')} / {safe_get(row, 'Department - Sub')}",
                'systems': f"{safe_get(row, 'System')} / {safe_get(row, 'System.1')}",
                'symptoms_complications': f"{safe_get(row, 'Disease Name')}, {safe_get(row, 'Disease Name.1')}",
                'gender': safe_get(row, 'Gender'),
                'age': safe_get(row, 'Age'),
                'bmi': safe_get(row, 'BMI')
            }
            patients_data.append(item)

        return jsonify({
            'patients': patients_data,
            'page': page,
            'total_pages': total_pages
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # CSV 파일 읽기
        df = pd.read_csv(file)

        # 전역 상태 업데이트
        current_data['df'] = df
        current_data['current_row'] = 0
        current_data['total_rows'] = len(df)

        return jsonify({
            'status': 'success',
            'total_rows': len(df)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/process_row', methods=['POST'])
def process_row():
    try:
        row_idx = request.json.get('row', 0)
        if current_data['df'] is None:
            return jsonify({'error': 'No data loaded'}), 400

        if row_idx >= len(current_data['df']):
            return jsonify({'error': 'Row index out of range'}), 400

        # 디버깅을 위한 출력 추가
        print(f"Processing row {row_idx}")
        print("Current data columns:", current_data['df'].columns.tolist())

        # 새로운 환자 분석 시작시 메모리 초기화
        processor.clear_memories()

        # 현재 행 데이터 추출 및 프롬프트 생성
        item = current_data['df'].iloc[row_idx]
        print("Row data:", item.to_dict())  # 행 데이터 출력

        input_data = prepare_input_data(item)
        print("Prepared input data:", input_data)  # 전처리된 데이터 출력

        # Stage 1
        stage1_prompt = STAGE_1_TEMPLATE.format(**input_data)
        print("Stage 1 prompt:", stage1_prompt)  # Stage 1 프롬프트 출력
        stage1_output = processor.process_single_prompt(stage1_prompt, processor.stage1_memory)

        # Stage 2
        stage2_output = processor.process_single_prompt(STAGE_2_TEMPLATE, processor.stage2_memory)

        # Stage 3
        stage3_prompt = STAGE_3_TEMPLATE.format(top_item=input_data['top_item'])
        stage3_output = processor.process_single_prompt(stage3_prompt, processor.stage3_memory)

        return jsonify({
            'status': 'success',
            'stage1_prompt': stage1_prompt,
            'stage1_output': stage1_output,
            'stage2_prompt': STAGE_2_TEMPLATE,
            'stage2_output': stage2_output,
            'stage3_prompt': stage3_prompt,
            'stage3_output': stage3_output
        })

    except Exception as e:
        print("=== Error Occurred ===")
        traceback.print_exc()
        print("====================")
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/process_stage', methods=['POST'])
def process_stage():
    try:
        row_idx = request.json.get('row', 0)
        stage = request.json.get('stage', 1)

        row_idx = int(row_idx)
        stage = int(stage)

        if current_data['df'] is None:
            return jsonify({'error': 'No data loaded'}), 400

        if row_idx >= len(current_data['df']):
            return jsonify({'error': 'Row index out of range'}), 400

        item = current_data['df'].iloc[row_idx]
        input_data = prepare_input_data(item)

        if stage == 1:
            # Stage 1은 새로운 환자 데이터로 시작하므로 메모리 초기화
            processor.clear_memories()
            prompt = STAGE_1_TEMPLATE.format(**input_data)
            output = processor.process_single_prompt(prompt, processor.stage1_memory)
            output_ko = translate_to_korean(output)

            return jsonify({
                'status': 'success',
                'stage': stage,
                'prompt': prompt,
                'output': output,
                'output_ko': output_ko
            })

        elif stage == 2:
            # Stage 1의 메모리를 가져와서 Stage 2 프롬프트와 결합
            memory_variables = processor.stage1_memory.load_memory_variables({})
            stage1_history = memory_variables.get("history", "")

            prompt = f"""Previous Analysis:
{stage1_history}

Based on this analysis and patient condition:
Disease: {input_data['disease_classification']}
Symptoms: {input_data['symptoms_complications']}

{STAGE_2_TEMPLATE}"""
            output = processor.process_single_prompt(prompt, processor.stage2_memory)
            output_ko = translate_to_korean(output)

            return jsonify({
                'status': 'success',
                'stage': stage,
                'prompt': prompt,
                'output': output,
                'output_ko': output_ko
            })

        elif stage == 3:
            # Stage 1과 2의 메모리를 모두 가져와서 Stage 3 프롬프트와 결합
            stage1_vars = processor.stage1_memory.load_memory_variables({})
            stage2_vars = processor.stage2_memory.load_memory_variables({})

            stage1_history = stage1_vars.get("history", "")
            stage2_history = stage2_vars.get("history", "")

            prompt = f"""Patient Analysis:
{stage1_history}

Management Plan:
{stage2_history}

Now, {STAGE_3_TEMPLATE.format(**input_data)}"""
            output = processor.process_single_prompt(prompt, processor.stage3_memory)
            output_ko = translate_to_korean(output)

            # Stage 3에서만 이미지 매칭 수행
            extracted_keywords = extract_keywords_from_output(output_ko)
            print("Extracted keywords:", extracted_keywords)

            # TF-IDF 매칭
            results = find_best_matches_tfidf(
                extracted_keywords,
                tfidf_vectorizer,
                tfidf_matrix,
                global_content_list,
                top_n=5
            )

            matched_images = []
            for score, item in results:
                if score > 0:
                    # 파일명에 .png 확장자 추가
                    filename = os.path.basename(item["파일명"])
                    if not filename.lower().endswith('.png'):
                        filename = f"{filename}.png"
                    matched_images.append({
                        'filename': filename,
                        'title': item["제목"]  # CSV에서 제목 정보 포함
                    })

            return jsonify({
                'status': 'success',
                'stage': stage,
                'prompt': prompt,
                'output': output,
                'output_ko': output_ko,
                'recommended_images': matched_images  # 파일명과 제목 함께 전달
            })

    except Exception as e:
        print("=== Error Occurred ===")
        traceback.print_exc()
        print("====================")
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


def init_tfidf_index(csv_path):
    global tfidf_vectorizer, tfidf_matrix, global_content_list
    # CSV 파일 로드
    global_content_list = load_content_csv(csv_path)
    # TF-IDF 인덱스 구축
    tfidf_vectorizer, tfidf_matrix, _ = build_tfidf_index(global_content_list)


def prepare_input_data(item):
    """데이터 전처리 함수"""
    try:
        disease_classification = ", ".join(filter(None, [
            safe_get(item, "Disease Classification - Primary"),
            safe_get(item, "Disease Classification - Secondary"),
            safe_get(item, "Disease Classification - Tertiary")
        ]))

        departments = ", ".join(filter(None, [
            safe_get(item, "Department - Main"),
            safe_get(item, "Department - Sub")
        ]))

        systems = ", ".join(filter(None, [
            safe_get(item, "System"),
            safe_get(item, "System.1"),
            safe_get(item, "System.2"),
            safe_get(item, "System.3"),
            safe_get(item, "System.4")
        ]))

        symptoms_complications = ", ".join(filter(None, [
            safe_get(item, "Disease Name"),
            safe_get(item, "Disease Name.1"),
            safe_get(item, "Disease Name.2"),
            safe_get(item, "Disease Name.3"),
            safe_get(item, "Disease Name.4")
        ]))

        # top-1(ENG) 값을 가져올 때 에러 처리 추가
        try:
            top_item = safe_get(item, "top-1(ENG)")
            if not top_item:  # 값이 없으면 기본값 설정
                top_item = "Diabetes management"
        except Exception as e:
            print(f"Error getting top-1(ENG): {e}")
            top_item = "Diabetes management"

        input_data = {
            'disease_classification': disease_classification,
            'departments': departments,
            'systems': systems,
            'symptoms_complications': symptoms_complications,
            'gender': safe_get(item, "Gender"),
            'age': safe_get(item, "Age"),
            'bmi': safe_get(item, "BMI"),
            'top_item': top_item
        }

        print("Created input_data:", input_data)  # 디버깅 출력
        return input_data

    except Exception as e:
        print(f"Error in prepare_input_data: {e}")
        raise

def setup_ngrok():
    # ngrok 설정 파일에서 인증 토큰과 도메인 읽기
    auth_token, domain = load_ngrok_config()

    if auth_token:
        ngrok.set_auth_token(auth_token)

    # 도메인이 있는 경우 해당 도메인으로 터널 생성
    if domain:
        tunnel = ngrok.connect(addr="5000", domain=domain)
    else:
        tunnel = ngrok.connect(5000)

    print(f' * ngrok 터널 URL: {tunnel.public_url}')
    return tunnel


if __name__ == '__main__':
    # init_tfidf_index("/content/drive/MyDrive/Colab Notebooks/llm4med/Graph Neural Network/recommendation.csv")
    init_tfidf_index(RECOMMENDATION_PATH)
    setup_ngrok()
    app.run(debug=False, host='0.0.0.0', port=5000)