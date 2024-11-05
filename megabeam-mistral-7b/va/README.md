This example is an end-to-end voice-assistant (VA) powered by the [amazon/MegaBeam-Mistral-7B-300k](https://huggingface.co/amazon/MegaBeam-Mistral-7B-300k) open LLM.

```
+------+     +--------+     +-----+     +-----+     +-----+     +--------+
|      |     |        |     |     |     |     |     |     |     |        |
| User | --> | Speech | --> | ASR | --> | LLM | --> | TTS | --> | Speech |
|      |     |        |     |     |     |     |     |     |     |        |
+------+     +--------+     +-----+     +-----+     +-----+     +--------+
   ^                                                                 |
   |                                                                 |
   +-----------------------------------------------------------------+
```
## Clone this git repository
```bash
git clone https://github.com/awslabs/extending-the-context-length-of-open-source-llms.git
cd extending-the-context-length-of-open-source-llms/megabeam-mistral-7b/va
```

## Setup instructions
Choose ONE of the two environments to set up Python

#### Create Conda environment
```bash
conda create -n megabeam-va-demo python=3.10
conda activate megabeam-va-demo
```

#### Create Virtual environment
```bash
python -m venv ~/venv/megabeam-va-demo
source ~/venv/megabeam-va-demo/bin/activate
```

### Install libraries
```bash
pip install -r requirements.txt

cd
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
pip install -e .
python -m unidic download
```

### Download models
```bash
export DOWNLOAD_DIR=~/Downloads

# download the quantized Amazon open LLM - MegaBeam-Mistral-7B-300K - 2.6GiB
wget -O ${DOWNLOAD_DIR}/MegaBeam-Mistral-7B-300k.Q2_K.gguf https://huggingface.co/RichardErkhov/amazon_-_MegaBeam-Mistral-7B-300k-gguf/resolve/main/MegaBeam-Mistral-7B-300k.Q2_K.gguf?download=true

# download the ASR model - 282 MiB
huggingface-cli download openai/whisper-base.en

# download the TTS model - 206 MiB
huggingface-cli download myshell-ai/MeloTTS-English
```

### Setup local LLM serving
1. Download and install [ollama](https://ollama.com/download)

```bash
export DOWNLOAD_DIR=~/Downloads
echo  "from ${DOWNLOAD_DIR}/MegaBeam-Mistral-7B-300k.Q2_K.gguf" > Modelfile
ollama create megabeam300k -f Modelfile
ollama run megabeam300k
```

## Run 
```bash
cd ~/extending-the-context-length-of-open-source-llms/megabeam-mistral-7b/va
chainlit run chainlit_app.py
```