

from transformers.modeling_gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer

from run_pplm_new import generate_text_pplm
import torch
import numpy as np
import json
from typing import List, Optional, Tuple, Union
from pplm_classification_head import ClassificationHead
from transformers.file_utils import cached_path
import time



pretrained_model="DialoGPT-medium"
discrim_weights='SST_classifier_head-3.pt'
discrim_meta_file='SST_classifier_head_meta-3.json'



cond_text="how are your typing skills ? <|endoftext|>"
class_label='negative'
num_samples=3




PPLM_DISCRIM = 2
loss_type = PPLM_DISCRIM

def get_classifier(
        discrim_meta: Optional[dict],
        class_label: Union[str, int],
        device: str
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if discrim_meta is None:
        return None, None

    params = discrim_meta
    classifier = ClassificationHead(
        class_size=params['class_size'],
        embed_size=params['embed_size']
    ).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified "
                         "in the discriminator model parameters")
    classifier.load_state_dict(
        torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]

    else:
        label_id = params["default_class"]

    return classifier, label_id



# set Random seed
seed=0
torch.manual_seed(seed)
np.random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# load pretrained model
model = GPT2LMHeadModel.from_pretrained(pretrained_model,output_hidden_states=True)

# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

model.to(device)
model.eval()

# Freeze GPT-2 weights
for param in model.parameters():
    param.requires_grad = False

with open(discrim_meta_file, 'r') as f:
    discrim_meta = json.load(f)
discrim_meta['path'] = discrim_weights

assert discrim_meta['pretrained_model']==pretrained_model






tokenized_cond_text = tokenizer.encode(tokenizer.bos_token + cond_text,add_special_tokens=False)


classifier, class_id = get_classifier(
    discrim_meta,
    class_label,
    device
)






start = time.time()
unpert_gen_tok_text, _, _ = generate_text_pplm(
    model=model,
    tokenizer=tokenizer,
    context=tokenized_cond_text,
    device=device,
    length=30,
    sample=True,
    perturb=False
)
end = time.time()


# untokenize unperturbed text
unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0])

print("= Unperturbed generated text =")
print(unpert_gen_text)
print()




if device == 'cuda':
    torch.cuda.empty_cache()





pert_gen_tok_texts = []
discrim_losses = []
losses_in_time = []


start = time.time()
for i in range(num_samples):
    pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=tokenized_cond_text,
        device=device,
        perturb=True,
        classifier=classifier,
        class_label=class_id,
        loss_type=loss_type,
        length=30,
        stepsize=0.05,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1,
        gm_scale=0.8,
        kl_scale=0.02
    )
    pert_gen_tok_texts.append(pert_gen_tok_text)
    if classifier is not None:
        discrim_losses.append(discrim_loss.data.cpu().numpy())
    losses_in_time.append(loss_in_time)
end = time.time()



if device == 'cuda':
    torch.cuda.empty_cache()

generated_texts = []

# iterate through the perturbed texts
for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
    try:
        # untokenize unperturbed text
        pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0])

        print("= Perturbed generated text {} =".format(i + 1))
        print(pert_gen_text)
        print()
    except:
        pass

    # keep the prefix, perturbed seq, original seq for each index
    generated_texts.append(
        (tokenized_cond_text, pert_gen_tok_text, unpert_gen_tok_text)
    )

