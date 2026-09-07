# A Text to Speech web app based on machine learning models

# https://daniel-chermetz.github.io/text-to-speech-matcha-with-hifi-vocder/

Try in the browser (no download) through the link above

Based on a partial WebGPU port of the Matcha-TTS and HiFi vocoder projects (both MIT licensed).

The following files were human written (100% manually written line-by-line):

     199 ./hifi-vocoder-cuda/inference_generator.js
     355 ./inference/inference_flow_matching.js
      21 ./inference/inference_time_embedding.js
      64 ./inference/inference_duration.js
     197 ./inference/inference_transformer.js
      30 ./inference/inference_flow_matching_transformer.js
     127 ./inference/blas_gemm.js
     214 ./inference/inference_pre_conv.js

1207 lines of code human written in total (~22% of the project)

The rest of the project was wholly written by AI (I'm very grateful for AI's help and contribution - the project wouldn't have been possible without it):

     368 ./index.html
     117 ./hifi-vocoder-cuda/generator_weights.mjs
     500 ./hifi-vocoder-cuda/inference_generator_webgpu.mjs
     112 ./web/matcha_tokenizer.mjs
      94 ./web/text_segmentation.mjs
      28 ./web/wav.mjs
     349 ./web/app.mjs
     379 ./web/reader.mjs
     190 ./web/matcha_phonemize.mjs
      82 ./single_sentence.html
      51 ./inference/flow_transformer_weights.mjs
     178 ./inference/inference_time_embedding_webgpu.mjs
      74 ./inference/text_encoder_webgpu.mjs
      68 ./inference/text_conditioning_webgpu.mjs
     105 ./inference/pre_conv_weights.mjs
     282 ./inference/inference_flow_matching_webgpu.mjs
     302 ./inference/inference_pre_conv_webgpu.mjs
      76 ./inference/inference_webgpu.mjs
      37 ./inference/duration_weights.mjs
     244 ./inference/inference_duration_webgpu.mjs
      31 ./inference/time_embedding_weights.mjs
     256 ./inference/inference_flow_matching_transformer_webgpu.mjs
      21 ./inference/pre_conv_example.mjs
      52 ./inference/flow_matching_weights.mjs
     288 ./inference/inference_transformer_webgpu.mjs
      59 ./inference/transformer_weights.mjs

4343 lines of code AI written in total (~78% of the project)
