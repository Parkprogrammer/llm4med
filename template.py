STAGE_1_TEMPLATE = """Focusing only on the current patient case:

Patient Information:
- Disease Classification: {disease_classification}
- Related Departments: {departments}
- Affected Systems: {systems}
- Potential Symptoms/Complications: {symptoms_complications}
- General Characteristics: Gender ({gender}), Age ({age}), BMI ({bmi})

Provide a concise summary of:
1. Main health issues for this specific patient
2. Key areas requiring management

Keep your response focused on answering only the 2 questions above."""

STAGE_2_TEMPLATE = """Based on the patient analysis from Stage 1, propose specific management plans for the following areas:
1. Comprehensive disease management
2. Individual management strategies for each potential complication
3. Recommendations for lifestyle improvements
4. Regular check-up and monitoring plan

Please provide specific management approaches for each area."""

STAGE_3_TEMPLATE = """Based on the previous analysis of the patient's condition and management plan,

and more specifically regarding the patients with
- Disease Classification: {disease_classification}
- Related Departments: {departments}
- Affected Systems: {systems}
- Potential Symptoms/Complications: {symptoms_complications}
- General Characteristics: Gender ({gender}), Age ({age}), BMI ({bmi})

please provide:

1. A specific list of 20 related topic keywords that the patient should focus on, specifically regarding {top_item}.
2. For each topic regarding {top_item}, include clear and practical advice that the patient can immediately implement to help with {top_item}.
3. The recommendations should be directly related to the patient's specific condition and circumstances.

Format your response as a numbered list of specific, actionable advice for the patient."""

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>iKooB 인공지능 기반 의료 데이터 분석 및 추천 시스템</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
    .chat-message {
        margin-bottom: 1.5rem;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .user-message {
        background-color: #ebf8ff;
        margin-left: 1rem;
        margin-right: 3rem;
        border-top-left-radius: 0.5rem;
    }
    .model-message {
        background-color: #ffffff;
        margin-left: 3rem;
        margin-right: 1rem;
        border-top-right-radius: 0.5rem;
    }
    .progress-circle {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        background: conic-gradient(#4CAF50 var(--progress), #f3f4f6 0deg);
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .progress-circle::before {
        content: attr(data-progress) '%';
        width: 50px;
        height: 50px;
        background: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.875rem;
        font-weight: bold;
        color: #4CAF50;
    }
    .card {
        background: white;
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .image-loading {
        position: relative;
        min-height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: #f3f4f6;
    }

    .image-loading::after {
        content: '이미지 로딩 중...';
        color: #6b7280;
    }
    </style>
<head>
    <title>iKooB 인공지능 기반 의료 데이터 분석 및 추천 시스템</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
    .chat-message {
        margin-bottom: 1.5rem;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .upload-zone-active {
        border-color: #3B82F6;
        background-color: #EFF6FF;
    }

    .table-row-hover:hover {
        background-color: #F9FAFB;
    }

    .analyze-btn {
        @apply px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200;
    }
    </style>
</head>
<body class="bg-gray-50">
    <div class="min-h-screen flex flex-col justify-center py-12">
        <div class="mx-auto w-full max-w-4xl px-4">
            <!-- 로고와 타이틀 -->
            <h1 class="text-4xl font-bold text-center mb-12 text-gray-800">
                iKooB 인공지능 기반<br>
                의료 데이터 분석 및 추천 시스템
            </h1>

            <!-- 파일 업로드 카드 -->
            <div class="bg-white rounded-xl shadow-lg p-8 mb-8" id="uploadSection">
                <form id="uploadForm" class="space-y-6">
                    <!-- 파일 업로드 영역 -->
                    <div class="upload-zone relative h-64 border-2 border-dashed border-gray-300 rounded-xl bg-gray-50 transition-all duration-200">
                        <input type="file"
                               accept=".csv"
                               id="fileInput"
                               name="file"
                               class="absolute inset-0 w-full h-full opacity-0 z-10 cursor-pointer"/>

                        <div class="absolute inset-0 flex flex-col items-center justify-center p-6">
                            <!-- 아이콘 -->
                            <svg class="w-16 h-16 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round"
                                      stroke-linejoin="round"
                                      stroke-width="2"
                                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
                            </svg>
                            <!-- 안내 텍스트 -->
                            <p class="text-lg font-medium text-gray-700 mb-2 text-center">
                                CSV 파일을 여기에 드롭하세요
                            </p>
                            <p class="text-sm text-gray-500 text-center" id="fileStatus">
                                또는 클릭하여 파일을 선택하세요
                            </p>
                        </div>
                    </div>

                    <!-- 업로드 버튼 -->
                    <button type="submit"
                            class="w-full py-4 bg-blue-600 text-white text-lg font-semibold rounded-lg hover:bg-blue-700 transition-colors duration-200">
                        분석 시작
                    </button>
                </form>
            </div>

            <!-- 환자 목록 (초기에는 숨김) -->
            <div class="bg-white rounded-xl shadow-lg p-8 mb-8 hidden" id="patientListContainer">
                <div class="flex items-center justify-between mb-6">
                    <h2 class="text-2xl font-bold text-gray-800">환자 목록</h2>
                    <div class="flex items-center space-x-4">
                        <button id="prevPageBtn" class="px-4 py-2 bg-gray-100 rounded-lg text-gray-700 hover:bg-gray-200 disabled:opacity-50">
                            이전
                        </button>
                        <span id="pageInfo" class="text-gray-600 font-medium"></span>
                        <button id="nextPageBtn" class="px-4 py-2 bg-gray-100 rounded-lg text-gray-700 hover:bg-gray-200 disabled:opacity-50">
                            다음
                        </button>
                    </div>
                </div>

                <div class="overflow-x-auto">
                    <table class="min-w-full divide-y divide-gray-200">
                        <thead class="bg-gray-50">
                            <tr>
                                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">No.</th>
                                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">질환 분류</th>
                                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">진료과</th>
                                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">시스템</th>
                                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">성별</th>
                                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">연령</th>
                                <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">분석</th>
                            </tr>
                        </thead>
                        <tbody id="patientListTable" class="bg-white divide-y divide-gray-200">
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- 대화형 프롬프트 표시 -->
            <div class="bg-white rounded-xl shadow-lg p-8 mb-8 hidden" id="chatContainer">
                <div class="flex items-center justify-between mb-4">
                    <h2 class="text-2xl font-bold text-gray-800">분석 결과</h2>
                    <span class="text-sm text-gray-500" id="analysisTime"></span>
                </div>
                <div id="chatMessages" class="space-y-4 max-h-[70vh] overflow-y-auto p-4 bg-gray-50 rounded-lg">
                </div>
            </div>

            <!-- 추천 이미지 표시 -->
            <div class="bg-white rounded-xl shadow-lg p-8 mb-8 hidden" id="recommendedImages">
                <h3 class="text-xl font-semibold mb-4 text-gray-700">추천 교육자료</h3>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4" id="imageGrid">
                </div>
                <p class="text-sm text-gray-600 mt-4 text-center">위는 아이쿱-AI가 추천해주는 교육자료입니다.</p>
            </div>

            <!-- 목록으로 돌아가기 버튼 -->
            <div class="fixed bottom-8 right-8 hidden" id="nextButtonContainer">
                <button id="nextRowButton" class="bg-blue-600 text-white rounded-full p-4 shadow-lg hover:bg-blue-700 transition-colors">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 5l7 7-7 7M5 5l7 7-7 7"/>
                    </svg>
                </button>
            </div>
        </div>
    </div>

    <script>

    let currentPage = 1;
    let totalPages = 1;
    let currentRow = 0;
    let totalRows = 0;


    marked.setOptions({
        breaks: true,
        gfm: true,
        pedantic: false,
        smartLists: true,
        smartypants: true
    });

    // Markdown 스타일 추가
    // 마크다운 스타일 업데이트
    const markdownStyles = `
    <style>
    .markdown-content {
        font-size: 1rem;
        line-height: 1.6;
    }

    .markdown-content p {
        margin-bottom: 1em;
        white-space: pre-line;  /* 줄바꿈만 유지 */
    }

    .markdown-content ul {
        list-style-type: disc;
        padding-left: 1.5em;
        margin-bottom: 1em;
    }

    .markdown-content ol {
        list-style-type: decimal;
        padding-left: 1.5em;
        margin-bottom: 1em;
    }

    .markdown-content li {
        margin: 0.5em 0;
        padding-left: 0.5em;
    }

    .markdown-content strong {
        font-weight: 600;
    }

    .markdown-content h1,
    .markdown-content h2,
    .markdown-content h3,
    .markdown-content h4 {
        margin: 1em 0 0.5em 0;
        font-weight: 600;
    }

    .markdown-content h1 { font-size: 1.5em; }
    .markdown-content h2 { font-size: 1.3em; }
    .markdown-content h3 { font-size: 1.2em; }
    .markdown-content h4 { font-size: 1.1em; }

    /* 코드 블록 스타일 */
    .markdown-content pre {
        background-color: #f6f8fa;
        padding: 1em;
        border-radius: 4px;
        overflow-x: auto;
        margin: 1em 0;
    }

    .markdown-content code {
        font-family: monospace;
        background-color: #f6f8fa;
        padding: 0.2em 0.4em;
        border-radius: 3px;
    }

    /* 인용문 스타일 */
    .markdown-content blockquote {
        border-left: 4px solid #ddd;
        padding-left: 1em;
        margin: 1em 0;
        color: #666;
    }

    /* 표 스타일 */
    .markdown-content table {
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
    }

    .markdown-content th,
    .markdown-content td {
        border: 1px solid #ddd;
        padding: 0.5em;
    }

    .markdown-content th {
        background-color: #f6f8fa;
    }
    </style>
    `;



    async function loadPatientList(page=1) {
      try {
          const res = await fetch(`/patients_list?page=${page}`);
          if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            alert("목록 로딩 실패: " + (errorData.error || res.statusText));
            return;
          }
          const data = await res.json();
          if (!data.patients) {
            data.patients = [];
          }


          // data.patients, data.page, data.total_pages
          currentPage = data.page;
          totalPages = data.total_pages;

          // 페이지 정보 갱신
          document.getElementById('pageInfo').textContent =
            `페이지 ${currentPage} / ${totalPages}`;

          // 이전/다음 버튼 상태
          document.getElementById('prevPageBtn').disabled = (currentPage <= 1);
          document.getElementById('nextPageBtn').disabled = (currentPage >= totalPages);

          // 테이블 내용 채우기
          const tbody = document.getElementById('patientListTable');
          tbody.innerHTML = '';

          data.patients.forEach(patient => {
              // tr
              const tr = document.createElement('tr');

              tr.innerHTML = `
                  <td class="px-4 py-2">${patient.index}</td>
                  <td class="px-4 py-2">${patient.disease_classification}</td>
                  <td class="px-4 py-2">${patient.departments}</td>
                  <td class="px-4 py-2">${patient.systems}</td>
                  <td class="px-4 py-2">${patient.gender}</td>
                  <td class="px-4 py-2">${patient.age}</td>
                  <td class="px-4 py-2">
                      <button class="bg-blue-500 text-white px-3 py-1 rounded analyzeBtn"
                              data-rowindex="${patient.index}">
                          분석하기
                      </button>
                  </td>
              `;
              tbody.appendChild(tr);
          });

          // "분석하기" 버튼 이벤트 바인딩
          document.querySelectorAll('.analyzeBtn').forEach(btn => {
              btn.addEventListener('click', (e) => {
                  const rowIndex = e.target.dataset.rowindex;
                  // 여기서 "환자 목록 섹션" 감추고,
                  // "대화창" 섹션 보이게 한 뒤,
                  // rowIndex를 가지고 /process_stage(stage=1) ~ stage=3 호출 (기존 로직)

                  document.getElementById('patientListContainer').classList.add('hidden');
                  document.getElementById('chatContainer').classList.remove('hidden');
                  document.getElementById('nextButtonContainer').classList.remove('hidden');

                  // 행 인덱스를 currentRow에 넣고
                  currentRow = rowIndex;
                  // 기존 processNextRow() 대신 "특정 rowIndex"를 처리하는 로직 호출
                  processStagesForRow(rowIndex);
              });
          });

      } catch (error) {
          console.error(error);
          alert("환자 목록 로딩 중 오류: " + error.message);
      }
  }


    function addMessage(content, isUser = true) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${isUser ? 'user-message' : 'model-message'}`;

        let parsedContent;
        if (isUser) {
            // 사용자 메시지는 그대로 표시
            parsedContent = content;
            messageDiv.innerHTML = `
                <div class="flex items-start">
                    <div class="flex-1">
                        <div class="whitespace-pre-wrap text-gray-800">${parsedContent}</div>
                    </div>
                </div>
            `;
        } else {
            // AI 응답은 마크다운 파싱
            parsedContent = marked.parse(content);
            messageDiv.innerHTML = `
                <div class="flex items-start">
                    <div class="flex-1">
                        <div class="markdown-content text-gray-800">${parsedContent}</div>
                    </div>
                </div>
            `;
        }

        chatMessages.appendChild(messageDiv);
        messageDiv.scrollIntoView({ behavior: 'smooth' });
    }


    function showRecommendedImages(imageUrls) {
      const container = document.getElementById('recommendedImages');
      const imageGrid = document.getElementById('imageGrid');
      imageGrid.innerHTML = ''; // 기존 이미지 클리어

      imageUrls.forEach(image => {
          const imgDiv = document.createElement('div');
          imgDiv.className = 'relative rounded-lg overflow-hidden shadow-lg cursor-pointer';
          imgDiv.innerHTML = `
              <img src="/image/${encodeURIComponent(image.filename)}"
                  alt="${image.title}"
                  class="w-full h-48 object-cover hover:opacity-75 transition-opacity"
                  onerror="this.src='https://via.placeholder.com/400x300?text=이미지+로드+실패'"
              />
              <div class="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white p-2 text-sm">
                  ${image.title}
              </div>
          `;

          // 이미지 클릭 이벤트 추가
          imgDiv.addEventListener('click', () => {
              showFullImage(image.filename, image.title);
          });

          imageGrid.appendChild(imgDiv);
      });

      container.classList.remove('hidden');
  }

  // 전체화면 이미지 모달 생성 및 표시
  function showFullImage(filename, title) {
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50';
    modal.innerHTML = `
        <div class="relative max-w-4xl max-h-screen p-4">
            <button class="absolute top-4 right-4 text-white text-xl font-bold z-10">&times;</button>
            <img src="/image/${encodeURIComponent(filename)}"
                 alt="${title}"
                 class="max-h-[90vh] max-w-full object-contain"
            />
            <div class="text-white text-center mt-2">${title}</div>
        </div>
    `;

    // 모달 닫기 이벤트
    modal.addEventListener('click', (e) => {
        if (e.target === modal || e.target.tagName === 'BUTTON') {
            modal.remove();
        }
    });

    document.body.appendChild(modal);
}

    // 파일 업로드 UI 개선
    const dropZone = document.querySelector('.upload-zone');
const fileStatus = document.getElementById('fileStatus');
const fileInput = document.getElementById('fileInput');

// 드래그 오버 효과
['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.add('upload-zone-active');
    });
});

// 드래그 종료 효과
['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.remove('upload-zone-active');
    });
});

// 파일 선택 시 UI 업데이트
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        fileStatus.textContent = `선택된 파일: ${file.name}`;
        fileStatus.classList.add('text-blue-600', 'font-medium');
    } else {
        fileStatus.textContent = '또는 클릭하여 파일을 선택하세요';
        fileStatus.classList.remove('text-blue-600', 'font-medium');
    }
});

// 파일 업로드 폼 제출
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const file = fileInput.files[0];
    if (!file) {
        alert('파일을 선택해주세요.');
        return;
    }

    try {
        const formData = new FormData();
        formData.append('file', file);

        // 업로드 버튼 비활성화 및 로딩 표시
        const submitBtn = e.target.querySelector('button[type="submit"]');
        submitBtn.disabled = true;
        submitBtn.innerHTML = `
            <svg class="animate-spin h-5 w-5 mr-3 inline" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
            업로드 중...
        `;

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const result = await response.json();

        if (result.status === 'success') {
            // 업로드 섹션 숨기기
            document.getElementById('uploadSection').classList.add('hidden');

            // 환자 목록 섹션 표시 및 데이터 로드
            loadPatientList(1);
            document.getElementById('patientListContainer').classList.remove('hidden');
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('파일 업로드 중 오류가 발생했습니다: ' + error.message);
    } finally {
        // 버튼 상태 복구
        submitBtn.disabled = false;
        submitBtn.innerHTML = '분석 시작';
    }
});

    async function processStagesForRow(rowIndex) {
      try {
          // Stage 1
          addMessage("Stage 1: 환자 정보 분석 중...", true);
          let response = await fetch('/process_stage', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ row: rowIndex, stage: 1 })
          });
          let result = await response.json();
          addMessage(result.output_ko || "에러", false);

          // Stage 2
          await new Promise(r => setTimeout(r, 1000));
          addMessage("Stage 2: 관리 계획 수립 중...", true);
          response = await fetch('/process_stage', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ row: rowIndex, stage: 2 })
          });
          result = await response.json();
          addMessage(result.output_ko || "에러", false);

          // Stage 3
          await new Promise(r => setTimeout(r, 1000));
          addMessage("Stage 3: 세부 권장사항 생성 중...", true);
          response = await fetch('/process_stage', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ row: rowIndex, stage: 3 })
          });
          result = await response.json();
          addMessage(result.output_ko || "에러", false);

          // 추천 이미지/파일 있으면 표시
          if (result.recommended_images && result.recommended_images.length > 0) {
              showRecommendedImages(result.recommended_images);
          }

      } catch (err) {
          console.error(err);
          addMessage("오류 발생: " + err.message, false);
      }
  }

    document.getElementById('nextRowButton').addEventListener('click', () => {
      // 대화창 리셋
      document.getElementById('chatMessages').innerHTML = '';
      document.getElementById('recommendedImages').classList.add('hidden');
      document.getElementById('imageGrid').innerHTML = '';

      // 환자 목록 섹션 다시 표시
      document.getElementById('patientListContainer').classList.remove('hidden');
      document.getElementById('chatContainer').classList.add('hidden');
      document.getElementById('nextButtonContainer').classList.add('hidden');
  });

  document.getElementById('prevPageBtn').addEventListener('click', () => {
      if (currentPage > 1) {
          loadPatientList(currentPage - 1);
      }
  });
  document.getElementById('nextPageBtn').addEventListener('click', () => {
      if (currentPage < totalPages) {
          loadPatientList(currentPage + 1);
      }
  });


  document.head.insertAdjacentHTML('beforeend', markdownStyles);
    </script>
</body>
</html>
"""