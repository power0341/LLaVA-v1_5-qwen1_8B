# LLaVA v1.5 + QWen1.8B = 2B

here is the briefing: replaced the language model with [QWen1.8B](https://huggingface.co/Qwen/Qwen-1_8B), so that made it 2.15B, pretrained the mm_adapter one epoch then finetuned by [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) another two epochs, training took 86.5 hours in total(4 RTX3090), it's able to score 74.19 on [VQAv2 testdev set](https://eval.ai/web/challenges/challenge-page/830/overview) [output](https://evalai.s3.amazonaws.com/media/submission_files/submission_325221/ceda300c-d784-47bb-9b30-81084c9770d2.json), you can find the checkpoint [here](https://huggingface.co/power0341/llava-v1_5-mlp2x-336px-qwen1_8b), have a try(make sure to set `conv-mode` to `qwen` or `mpt`), though it often underperforms hilariously ... and lastly, the inference time is outrageously long(> 2 second), I haven't figured it out where it went wrong yet.
