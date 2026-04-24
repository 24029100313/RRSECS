import torch, imagesize, json, os
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.dist_helper import setup_distributed



# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


def convert_bbox(bbox_str, w, h):
    bbox = bbox_str.strip('[]').split(',')
    bbox = [float(coord) for coord in bbox]

    x_min = round(bbox[0] * w, 1)
    y_min = round(bbox[1] * h, 1)
    x_max = round(bbox[2] * w, 1)
    y_max = round(bbox[3] * h, 1)

    return [x_min, y_min, x_max, y_max]

### single
# def grounding(model, processor, img_path, expression):
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "image": img_path,
#                 },
#                 {
#                     "type": "text",
#                     "text": "Please provide the bounding box coordinate of the region this sentence describes: " + expression + "."},
#             ],
#         }
#     ]
#
#     # Preparation for inference
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     image_inputs, video_inputs = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to("cuda")
#
#     # Inference: Generation of the output
#     generated_ids = model.generate(**inputs, max_new_tokens=128)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
#
#     return output_text
#
#
# model_path = '/root/data3/FM/VLM/Qwen2-VL/Qwen2-VL-72B-Instruct'
# model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
# processor = AutoProcessor.from_pretrained(model_path)
#
# val_infos = json.load(open('/root/lxq/RSFM_RIS_Datasets/Optical/RefDIOR/phrase_txts/test.json'))
# results = []
#
# for k, v in tqdm(val_infos.items()):
#     result = {}
#     img_path = '/root/lxq/RSFM_RIS_Datasets/Optical/RefDIOR/images/' + k + '.jpg'
#     sentence = v['sentences'][0]
#
#     output = grounding(model, processor, img_path, sentence)
#     result[k] = output
#     results.append(result)
#
# json.dump(results, open('./qwen2vl_72b_refdiortest.json', 'w'))


### Multiple
def grounding(model, processor, messages):
    # Preparation for inference
    texts = [
        processor.apply_chat_template(mes, tokenize=False, add_generation_prompt=True)
        for mes in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_texts


results = []
model_path = '/root/data3/FM/VLM/Qwen2-VL/Qwen2-VL-2B-Instruct'
model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

val_infos = json.load(open('/root/lxq/RSFM_RIS_Datasets/Optical/RefDIOR/phrase_txts/val.json'))
inputs = []
for k, v in val_infos.items():
    inputs.append(('/root/lxq/RSFM_RIS_Datasets/Optical/RefDIOR/images/' + k + '.jpg',
                   v['sentences'][0]))

res_ss = [inputs[i:i+4] for i in range(0, len(inputs), 4)]
for res_s in tqdm(res_ss):
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": res[0],
                    },
                    {
                        "type": "text",
                        "text": "Please provide the bounding box coordinate of the region this sentence describes: " + res[1] + "."},
                ],
            }
        ] for res in res_s
    ]
    outputs = grounding(model, processor, messages)
    for res, output in zip(res_s, outputs):
        result = (res[0], output)
        results.append(result)

json.dump(results, open('./qwen2vl_72b_refdiorval.json', 'w'))









# class RefDIORDataset(Dataset):
#     def __init__(self, json_file, processor):
#         self.img_ids, self.expressions = self._load_json(json_file)
#         self.processor = processor
#
#     def __len__(self):
#         return len(self.img_ids)
#
#     def __getitem__(self, idx):
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image",
#                         "image": '/root/lxq/RSFM_RIS_Datasets/Optical/RefDIOR/images/' + self.img_ids[idx] + '.jpg',
#                     },
#                     {
#                         "type": "text",
#                         "text": "Please provide the bounding box coordinate of the region this sentence describes: " + self.expressions[idx] + "."},
#                 ],
#             }
#         ]
#
#         text = self.processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = self.processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )
#
#         return inputs
#
#     def _load_json(self, json_file):
#         img_ids, expressions = [], []
#         for k, v in json.load(open(json_file, 'r')).items():
#             img_id, expression = k, v['sentences'][0]
#             img_ids.append(img_id)
#             expressions.append(expression)
#
#         return img_ids, expressions
#
# setup_distributed()
#
# model_path = '/root/data3/FM/VLM/Qwen2-VL/Qwen2-VL-2B-Instruct'
# model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
# processor = AutoProcessor.from_pretrained(model_path)
#
# valset = RefDIORDataset('/root/lxq/RSFM_RIS_Datasets/Optical/RefDIOR/phrase_txts/test.json', processor)
# valsampler = torch.utils.data.distributed.DistributedSampler(valset)
# valloader = DataLoader(valset, batch_size=8, pin_memory=False,
#                        num_workers=8, drop_last=False, sampler=valsampler)
#
# with torch.no_grad():
#     for x in valloader:
#         inputs = x.to("cuda")
#
#         generated_ids = model.generate(**inputs, max_new_tokens=128)
#         generated_ids_trimmed = [
#             out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#         ]
#         output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
#                                              clean_up_tokenization_spaces=False)
#     print(output_text)