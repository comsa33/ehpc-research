# CUDA 베이스 이미지
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# 비대화형 모드 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 필요한 기본 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-nvcc-12-2 \
    cuda-cudart-dev-12-2 \
    cuda-command-line-tools-12-2 \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    git \
    ca-certificates \
    libpq-dev \
    fonts-nanum \
    fonts-noto-cjk \
    locales \
    fontconfig \
    fontconfig-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# libdevice.10.bc 파일 위치 설정
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-12.2
ENV TF_CUDA_PATHS=/usr/local/cuda-12.2

# TensorFlow와 CUDA 연동 최적화
ENV CUDA_VISIBLE_DEVICES=0
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# 한글 로케일 설정
RUN sed -i -e 's/# ko_KR.UTF-8 UTF-8/ko_KR.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG=ko_KR.UTF-8
ENV LC_ALL=ko_KR.UTF-8

# 폰트 캐시 갱신
RUN fc-cache -fv

# 작업 디렉토리 설정
WORKDIR /app

# 환경변수 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# UV 설치 (공식 이미지에서 바이너리만 복사)
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

# UV를 확인
RUN uv --version

# 의존성 파일 먼저 복사 (캐싱 최적화)
COPY pyproject.toml uv.lock /app/

# 소스코드 복사
COPY . .

# 의존성 설치
RUN uv sync --no-dev --frozen --no-cache

# 포트 설정
EXPOSE 8501

# 실행 명령어 설정 (pyenv 환경 활성화 포함)

# # 서버실행
CMD ["uv", "run", "streamlit", "run", "demo/streamlit_app.py"]