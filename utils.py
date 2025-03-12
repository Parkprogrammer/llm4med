import urllib
from urllib.error import HTTPError
import time
import json
import os
from pyngrok import ngrok
import json

# 프로토타입 테스팅을 하기 위한 ngrok config 파일 로드
# def load_ngrok_config():
#     try:
#         with open('/content/drive/MyDrive/Colab Notebooks/llm4med/Graph Neural Network/ngrok_config.json', 'r') as f:
#             config = json.load(f)
#             return config.get('authtoken'), config.get('domain')
#     except FileNotFoundError:
#         return None, None
def load_ngrok_config():
    """
    ngrok 설정 파일에서 인증 토큰과 도메인을 읽어오는 함수
    
    Returns:
        tuple: (auth_token, domain) - 인증 토큰과 도메인 정보
    """
    try:
        config_path = 'ngrok_config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                auth_token = config.get('auth_token', '')
                domain = config.get('domain', '')
                return auth_token, domain
        else:
            print(f"ngrok 설정 파일이 없습니다: {config_path}")
            # 환경 변수에서 확인
            auth_token = os.environ.get('NGROK_AUTH_TOKEN', '')
            domain = os.environ.get('NGROK_DOMAIN', '')
            return auth_token, domain
    except Exception as e:
        print(f"ngrok 설정 로드 중 오류: {e}")
        return '', ''

    
# TsinghuaC3I/Llama-3-8B-UltraMedical https://huggingface.co/TsinghuaC3I/Llama-3-8B-UltraMedical
# 위 모델은 Bilingual이 아닌 English-Native이기 때문에 해당 모델에 번역 preprocess를 진행해줌.
# TODO: PAPAGO API key를 설정해주어야 함.
def translate_to_korean(text, client_id=None, client_secret=None):
    """
    Papago API를 사용하여 영어를 한국어로 번역
    긴 텍스트는 여러 부분으로 나누어 번역
    """
    if not text:
        return ""

    try:
        max_chunk_size = 4000  # Papago API는 5000자 제한이 있으므로 여유있게 4000자로 설정
        chunks = []

        # 문단 단위로 텍스트 분할
        paragraphs = text.split('\n\n')
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < max_chunk_size:
                current_chunk += (paragraph + '\n\n')
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n\n'

        if current_chunk:
            chunks.append(current_chunk.strip())

        translated_chunks = []
        for chunk in chunks:
            try:
                url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
                enc_text = urllib.parse.quote(chunk)
                data = f"source=en&target=ko&text={enc_text}"

                request = urllib.request.Request(url)
                request.add_header("X-NCP-APIGW-API-KEY-ID", client_id)
                request.add_header("X-NCP-APIGW-API-KEY", client_secret)

                response = urllib.request.urlopen(request, data=data.encode("utf-8"))
                rescode = response.getcode()

                if rescode == 200:
                    response_body = response.read()
                    result = json.loads(response_body.decode("utf-8"))
                    translated_chunk = result['message']['result']['translatedText']
                    translated_chunks.append(translated_chunk)
                    print(f"청크 번역 성공 (길이: {len(chunk)})")
                else:
                    print(f"API 에러 (코드: {rescode}): {chunk[:100]}...")
                    translated_chunks.append(chunk)

            except Exception as e:
                print(f"청크 번역 중 오류 발생: {str(e)}")
                translated_chunks.append(chunk)

            time.sleep(0.5)

        final_translation = '\n\n'.join(translated_chunks)
        return final_translation

    except Exception as e:
        print(f"전체 번역 과정 중 오류 발생: {str(e)}")
        return text

# NOTE: 포트번호는 작업 환경 설정에 맞게 지정
def setup_ngrok(port=5000):
    """
    ngrok 터널을 설정하고 시작하는 함수
    
    Args:
        port (int): 로컬 서버 포트 번호
        
    Returns:
        tunnel: 생성된 ngrok 터널 객체
    """
    # ngrok 설정 파일에서 인증 토큰과 도메인 읽기
    auth_token, domain = load_ngrok_config()

    if auth_token:
        ngrok.set_auth_token(auth_token)

    # 도메인이 있는 경우 해당 도메인으로 터널 생성
    if domain:
        tunnel = ngrok.connect(addr=port, domain=domain)
    else:
        tunnel = ngrok.connect(port)

    print(f' * ngrok 터널 URL: {tunnel.public_url}')
    return tunnel

def close_ngrok():
    """
    모든 ngrok 터널을 종료하는 함수
    """
    try:
        ngrok.kill()
        print(" * ngrok 터널이 종료되었습니다.")
    except Exception as e:
        print(f" * ngrok 터널 종료 중 오류: {e}")