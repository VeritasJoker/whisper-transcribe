
CMD := echo
CMD := sbatch submit.sh
CMD := python

SID := 500
CONV_IDX := 798_30s_test

SID := 661
CONV_IDX := Podcast

# SID := 625
# CONV_IDX := $(shell seq 0 53)

# SID := 676
# CONV_IDX := $(shell seq 37 38)

# SID := 7170
# CONV_IDX := $(shell seq 1 24)

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