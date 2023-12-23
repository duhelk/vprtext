from Levenshtein import distance

_LANGS = ['en']


def calculate_ned(gt_texts, gt_langs, pred_texts, ned_file):
    gt_texts = [''.join(x.split(' '))  for x in gt_texts]
    pred_texts = [''.join(x.split(' '))  for x in pred_texts]
    gt_neds = dict.fromkeys(gt_texts, 1)
    gt_preds = dict.fromkeys(gt_texts, '')
    for gt, lang in zip(gt_texts, gt_langs):
        if lang not in _LANGS:
            print('Not evaluating', lang, gt)
            del gt_neds[gt]
            continue
        if len(gt) == 0:
            print('GT with 0 len found.')
            del gt_neds[gt]
            continue
        for txt in pred_texts:
            dist = distance(gt.lower(), txt.lower())
            max_len = max([len(gt), len(txt), 1])
            ned = dist/max_len
            if ned < gt_neds[gt]:
                gt_neds[gt] = ned
                gt_preds[gt] = txt
        with open(ned_file, "a") as file:
            file.write(f'{gt} , {lang}, {gt_preds[gt]} , {1-gt_neds[gt]}\n')
    return gt_neds
