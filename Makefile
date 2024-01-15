
CMD := echo
CMD := python
CMD := sbatch submit.sh

SID := 500
CONV_IDX := 798_30s_test

SID := 661
CONV_IDX := Podcast

SID := 625
CONV_IDX := $(shell seq 0 53)

SID := 676
CONV_IDX := $(shell seq 0 77)

SID := 7170
SID := 717
CONV_IDX := $(shell seq 0 23)

# SID := 798
# CONV_IDX := $(shell seq 0 14)



whisper-transcribe:
	for conv in $(CONV_IDX); do \
		$(CMD) scripts/whisper_transcribe.py \
			--sid $(SID) \
			--model large-v2 \
			--conv-idx $$conv \
			--time-stamp; \
	done;

whisperx-transcribe:
	for conv in $(CONV_IDX); do \
		$(CMD) scripts/whisperx_transcribe.py \
			--sid $(SID) \
			--model large-v2 \
			--conv-idx $$conv; \
	done;


whisperx-paper:
	$(CMD) scripts/whisperx_paper_trans.py \
		--sid $(SID) \
		--model large-v2 \


audio-erp:
	for conv in $(CONV_IDX); do \
		python scripts/evaluate_audio_erp.py \
			--sid $(SID) \
			--model large-v2 \
			--conv-idx $$conv; \
	done;


align-speaker:
	python scripts/align_speaker.py \
		--sid $(SID) \
		--model large-v2-x